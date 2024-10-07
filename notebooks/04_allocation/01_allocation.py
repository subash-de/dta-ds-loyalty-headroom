# Databricks notebook source
# MAGIC %run ../setup/bootstrap

# COMMAND ----------

from datetime import datetime, timedelta

import seaborn as sns
from dtaml.logging import get_logger
from pyspark.sql import Window as W
from pyspark.sql import functions as F

import customer_headroom.utils.persist_utils as persist_utils
from customer_headroom.allocation.allocator import Allocator
from customer_headroom.utils import tmo_utils

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


logger.info(
    f"""
config.dates: {config.dates}
campaign: {campaign}
last_registration_date: {last_registration_date}
"""
)

# COMMAND ----------


logger.info("Begin Allocation")
config_al = config["allocation"]
under_predict_adjustment_factor = config_al["headroom_factor"]

prediction_tbl_name = persist_utils.get_table_name(
    factory_database=config.factory_database,
    lab_database=config.lab_database,
    table_prefix=config_al.prediction_tbl.prefix,
    sensitivity=config.sensitivity,
)
logger.info(f"""prediction_tbl_name: {prediction_tbl_name}""")

predictions = persist_utils.read_table(
    table_name=prediction_tbl_name, where=f"campaign={campaign}"
)

# if its under predicting, then would force the stretch to be 20%
predictions = predictions.withColumn(
    "prediction_out_orig", F.lit(F.col("prediction_out"))
).withColumn(
    "prediction_out",
    F.when(
        F.col("prediction_out_orig") < F.col(config_al["feature_col"]),
        F.col(config_al["feature_col"]) * under_predict_adjustment_factor,
    ).otherwise(F.col("prediction_out_orig")),
)
predictions_cnt = predictions.count()
logger.info(f"""predictions_cnt: {predictions_cnt}""")

# spend and save feature column l2_id_total_spend_basket

# allocate for spend and save
if config_al["tcol_allocate_separately"]:
    # TCOL segment
    segtco_history_df = spark.table("customer_azbase_prod.segtco_history")
    segtco_history_ = tmo_utils.get_preceding_segtco_history(
        segtco_history_df, campaign
    )

    predictions = predictions.join(
        segtco_history_.select("cust_id", "cust_band_fd"), on="cust_id", how="left"
    )
    prediction_top = predictions.filter(F.col("cust_band_fd").contains("Top"))
    prediction_not_top = predictions.filter(~F.col("cust_band_fd").contains("Top"))
    assert (
        prediction_top.count() > 0
    ), f"No of rows for customer in top group, got {prediction_top.count()}"

    # top allocation
    logger.info(
        f"Allocation top customer, number of top customers {prediction_top.select('cust_id').distinct().count()}"
    )
    allocation_manager_top = Allocator(
        feature_col=config_al["feature_col"],
        offer_limits=config["offer_limits_top"],  # change this for new top offer
        offer_desc=config["offers_desc_top"],  # change this for new top offer
        user_key=config_al["user_key"],
        lx_key=config_al["lx_key"],
        outlier_min=config_al["outlier_min"],
        outlier_max=config_al["outlier_max"],
        max_increase=config_al["max_increase"],
        min_increase=config_al["min_increase"],
        headroom_factor=config_al["headroom_factor"],
        fill_offer=config_al["fill_offer"],  # change this for top customer
        prev_not_bought_factor=config_al["prev_not_bought_factor"],
    )
    headroom_export_top = allocation_manager_top.get(prediction_top).withColumn(
        "campaign", F.lit(campaign)
    )
    logger.info("Allocation top customer - finished")

    # non-top allocation
    logger.info(
        f"Allocation NOT top customer, number of none top customers {prediction_not_top.select('cust_id').distinct().count()}"
    )
    allocation_manager_not_top = Allocator(
        feature_col=config_al["feature_col"],
        offer_limits=config["offer_limits"],
        offer_desc=config["offers_desc"],
        user_key=config_al["user_key"],
        outlier_min=config_al["outlier_min"],
        outlier_max=config_al["outlier_max"],
        max_increase=config_al["max_increase"],
        min_increase=config_al["min_increase"],
        headroom_factor=config_al["headroom_factor"],
        fill_offer=config_al["fill_offer"],
        prev_not_bought_factor=config_al["prev_not_bought_factor"],
    )
    headroom_export_not_top = allocation_manager_not_top.get(
        prediction_not_top
    ).withColumn("campaign", F.lit(campaign))
    logger.info("Allocation NOT top customer - finished")

    headroom_export = headroom_export_top.union(headroom_export_not_top)

else:
    logger.info("Allocating all customers")
    allocation_manager = Allocator(
        feature_col=config_al["feature_col"],
        offer_limits=config["offer_limits"],
        offer_desc=config["offers_desc"],
        user_key=config_al["user_key"],
        lx_key=config_al["lx_key"],
        outlier_min=config_al["outlier_min"],
        outlier_max=config_al["outlier_max"],
        max_increase=config_al["max_increase"],
        min_increase=config_al["min_increase"],
        headroom_factor=config_al["headroom_factor"],
        fill_offer=config_al["fill_offer"],
        prev_not_bought_factor=config_al["prev_not_bought_factor"],
        prev_not_bought_factor_lx_id_indpendent=config_al[
            "prev_not_bought_factor_lx_id_indpendent"
        ],
        aggregate_level=config_al["aggregate_level"],
    )

    headroom_export = allocation_manager.get(predictions).withColumn(
        "campaign", F.lit(campaign)
    )
    
headroom_export_cnt = headroom_export.count()
logger.info(f"""headroom_export_cnt: {predictions_cnt}""")

stg_headroom_tbl_name = persist_utils.create_beam_table(
    table_prefix=config_al.headroom_export_tbl.prefix,
    lab_database=config.lab_database,
    factory_database=config.factory_stg_database,
    sensitivity=config.sensitivity,
    schema=headroom_export,
    partition_by=config_al.headroom_export_tbl.partitionByList,
    overwrite_table=True,
    assert_equality=False,
    add_load_timestamp=True,
)
logger.info(f"""stg_headroom_tbl_name: {stg_headroom_tbl_name}""")

persist_utils.insert_df_into_table(
    target_tbl_name=stg_headroom_tbl_name,
    insert_df=headroom_export,
    delete_where=f"campaign={campaign}",
    insert_append=True,
    add_columns=True,
)

if config_al["aggregate_level"] == "basket":
    if config["exclude_high_spend"] is not None:
        logger.info(
            f"Remove customer whos spend_plus_headroom > {config['exclude_high_spend']}"
        )
        headroom_export = headroom_export.filter(
            F.col("spend_plus_headroom") <= config["exclude_high_spend"]
        )

if config["min_num_basket"] is not None:
    logger.info(
        f"Remove customer who have less than {config['min_num_basket']} basket"
    )
    headroom_export = headroom_export.join(
        predictions.filter(F.col("count_user_basket") >= config["min_num_basket"])
        .select("cust_id")
        .distinct(),
        how="inner",
        on="cust_id",
    )

headroom_export_cnt = headroom_export.count()
logger.info(f"""headroom_export_cnt: {predictions_cnt}""")

headroom_tbl_name = persist_utils.create_beam_table(
    table_prefix=config_al.headroom_export_tbl.prefix,
    lab_database=config.lab_database,
    factory_database=config.factory_database,
    sensitivity=config.sensitivity,
    schema=headroom_export,
    partition_by=config_al.headroom_export_tbl.partitionByList,
    overwrite_table=True,
    assert_equality=False,
    add_load_timestamp=True,
)
logger.info(f"""headroom_tbl_name: {headroom_tbl_name}""")

persist_utils.insert_df_into_table(
    target_tbl_name=headroom_tbl_name,
    insert_df=headroom_export,
    delete_where=f"campaign={campaign}",
    insert_append=True,
    add_columns=True,
)


# COMMAND ----------


config_al = config["allocation"]
headroom_tbl_name = persist_utils.get_table_name(
    factory_database=config.factory_database,
    lab_database=config.lab_database,
    table_prefix=config_al.headroom_export_tbl.prefix,
    sensitivity=config.sensitivity,
)
logger.info(f"""headroom_tbl_name: {headroom_tbl_name}""")

headroom_tbl = persist_utils.read_table(
    table_name=headroom_tbl_name, where=f"campaign={campaign}"
)
display(headroom_tbl.orderBy(F.rand()))

# COMMAND ----------

headroom_tbl.groupBy("desc").count().withColumn(
    "percentage", F.round(F.col("count") / F.sum("count").over(W.partitionBy()), 3)
).display()

# COMMAND ----------

dbutils.notebook.exit(True)
