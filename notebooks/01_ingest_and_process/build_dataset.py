# Databricks notebook source
# MAGIC %run ./bootstrap

# COMMAND ----------

from ast import literal_eval
from datetime import datetime, timedelta

import seaborn as sns
from dtaml.logging import get_logger
from pyspark.sql import functions as F

import customer_headroom.utils.persist_utils as persist_utils
from customer_headroom.etl.build_dataset import TransactionsManager
from customer_headroom.etl.segmentation import (
    SegmentationDataManager,
    SegmentationManager,
)

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

logger.info(
    f"""
config_dates: {config_dates}
campaign: {campaign}
last_registration_date: {last_registration_date}
"""
)

# COMMAND ----------

# MAGIC %md # Build Segmentations

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

    # selecting customer who have registered for a period of time, so that there is some spending data when calculating percentiles.
    sparks_account_df = sparks_account_df.filter(
        F.col("registration_date")
        <= datetime.strptime(str(last_registration_date), "%Y%m%d")
    )

    custs_etl_data = sparks_account_df

    seg_data_manager = SegmentationDataManager(
        etl_date=get_date(config_sg["etl_date"]),
        lookback_days=config_sg["lookback_days"],
        l1_id=config_sg["l1_id"],
        user_id=config_sg["user_id"],
    )

    seg_data = seg_data_manager.get(
        trx_line_df, sparks_account_df, cust_master_df, customer_input=custs_etl_data
    ).withColumn("campaign", F.lit(campaign))

    # TODO: Replace with customer cluster work to reduce data sizes to appropiate groups.
    seg_data_table_name = persist_utils.create_beam_table(
        table_prefix=config_sg.seg_data_tbl.prefix,
        lab_database=config.dev_database,
        factory_database=config_sg.seg_data_tbl.factory_database,
        sensitivity=config_sg.seg_data_tbl.sensitivity,
        schema=seg_data,
        partition_by=config_sg.seg_data_tbl.partitionByList,
        overwrite_table=False,
        assert_equality=False,
        add_load_timestamp=True,
    )

    logger.info(f"""seg_data_table_name: {seg_data_table_name}""")

    persist_utils.insert_df_into_table(
        target_tbl_name=seg_data_table_name,
        insert_df=seg_data,
        delete_where=f"campaign={campaign}",
    )

    logger.info("Begin Segmentation of Dataset")
    seg_data_read = persist_utils.read_table(
        table_name=seg_data_table_name, where=f"campaign={campaign}"
    )
    seg_manager = SegmentationManager(
        num_cols=config_sg["num_cols"],
        cat_cols=config_sg["cat_cols"],
        frac_lim=config_sg["frac_lim"],
        fail_limit=config_sg["fail_limit"],
        k_search_min=config_sg["k_search_min"],
        k_search_max=config_sg["k_search_max"],
        pca_k=config_sg["pca_k"],
        data_lower_lim=config_sg["data_lower_lim"],
        user_id=config_sg["user_id"],
        verbose=config_sg["verbose"],
    )

    segmentations = seg_manager.get(data=seg_data_read).withColumn(
        "campaign", F.lit(campaign)
    )
    segmentations_tbl_name = persist_utils.create_beam_table(
        table_prefix=config_sg.segmentations_tbl.prefix,
        lab_database=config.dev_database,
        factory_database=config_sg.segmentations_tbl.factory_database,
        sensitivity=config_sg.segmentations_tbl.sensitivity,
        schema=segmentations,
        partition_by=config_sg.segmentations_tbl.partitionByList,
        overwrite_table=False,
        assert_equality=False,
        add_load_timestamp=True,
    )
    logger.info(f"""segmentations_tbl_name: {segmentations_tbl_name}""")

    persist_utils.insert_df_into_table(
        target_tbl_name=segmentations_tbl_name,
        insert_df=segmentations,
        delete_where=f"campaign={campaign}",
    )

# COMMAND ----------

# MAGIC %md # Get Segmentations to use

# COMMAND ----------

if any(step in config.steps for step in ("build_dataset", "fit_rec", "predict")):
    config_use = config["use_segments"]
    if config_use["all"] == True:
        logger.info("Use all Segmentations")
        # In
        segmentations_tbl_name = persist_utils.get_table_name(
            factory_database=config_use.segmentations_tbl.factory_database,
            lab_database=config.dev_database,
            table_prefix=config_use.segmentations_tbl.prefix,
            sensitivity=config_use.segmentations_tbl.sensitivity,
        )
        segmentations_tbl = persist_utils.read_table(
            table_name=segmentations_tbl_name, where=f"campaign={campaign}"
        )
        seg_list = find_all_segments(segmentations_tbl, config_use["partitionByList"])
    else:
        seg_list = config_use["seg_list"]
    logger.info(f"Segmentations: {seg_list}")

# COMMAND ----------

# MAGIC %md # Build dataset

# COMMAND ----------

config_use = config["use_segments"]
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

    segmentations_tbl_name = persist_utils.get_table_name(
        table_prefix=config_bd.segmentations_tbl.prefix,
        lab_database=config.dev_database,
        factory_database=config_bd.segmentations_tbl.factory_database,
        sensitivity=config_bd.segmentations_tbl.sensitivity,
    )
    logger.info(f"""segmentations_tbl_name: {segmentations_tbl_name}""")

    segmentations_tbl = persist_utils.read_table(
        table_name=segmentations_tbl_name, where=f"campaign= {campaign}"
    ).select([config_bd["user_id"]] + partitionByList)

    # build training data
    trx_manager = TransactionsManager(
        etl_date=get_date(config_bd["etl_date"]),
        lookback_days=config_bd["lookback_days"],
        l1_ids=config_bd["l1_ids"],
        lx=config_bd["lx"],
        lx_ids=config_bd["lx_ids"],  # getting all the products in this l2 id
        user_key=config_bd["user_id"],
        window_days=config_bd["window_days"],
        time_window_length=config_bd["time_window_days"],
        exclude_items=literal_eval(config["exclude_items"]),
        aggregation_level=config_bd["aggregation_level"],
    )
    all_data = trx_manager.get(trx_line_df, articles_df, cust_seg=segmentations_tbl)

    etl_data_tbl_name = persist_utils.create_beam_table(
        table_prefix=config_bd.etl_data_tbl.prefix,
        lab_database=config.dev_database,
        factory_database=config_bd.etl_data_tbl.factory_database,
        sensitivity=config_bd.etl_data_tbl.sensitivity,
        schema=all_data,
        partition_by=config_bd.etl_data_tbl.partitionByList,
        overwrite_table=False,
        assert_equality=False,
        add_load_timestamp=True,
    )
    logger.info(f"""etl_data_tbl_name: {etl_data_tbl_name}""")

    persist_utils.insert_df_into_table(
        target_tbl_name=etl_data_tbl_name,
        insert_df=all_data,
        add_columns=True,
        insert_append=True,
        delete_where=f"campaign={campaign}",
    )

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
    etl_data_tbl_name = persist_utils.get_table_name(
        factory_database=config_bd.etl_data_tbl.factory_database,
        lab_database=config.dev_database,
        table_prefix=config_bd.etl_data_tbl.prefix,
        sensitivity=config_bd.etl_data_tbl.sensitivity,
    )

    logger.info(f"""etl_data_tbl_name: {etl_data_tbl_name}""")

    etl_data_tbl = persist_utils.read_table(
        table_name=etl_data_tbl_name, where=f"campaign={campaign}"
    )
    display(etl_data_tbl.orderBy(F.rand()))

# COMMAND ----------


def get_count(seg, config, database):
    partitionByList = seg.keys()
    seg_ext = [f"({k}='{seg[k]}')" for k in partitionByList]

    etl_data_tbl_name = persist_utils.get_table_name(
        factory_database=config.etl_data_tbl.factory_database,
        lab_database=database,
        table_prefix=config.etl_data_tbl.prefix,
        sensitivity=config.etl_data_tbl.sensitivity,
    )

    seg_etl_data_tbl = persist_utils.read_table(
        table_name=etl_data_tbl_name, where=" and ".join(seg_ext)
    )

    cnt = seg_etl_data_tbl.count()

    return (seg, cnt)


seg_cnt = []
for seg in seg_list:
    config_bd = config["build_dataset"]
    seg_cnt.append(get_count(seg, config=config_bd, database=config.dev_database))

seg_cnt.sort(key=lambda i: i[1], reverse=True)

seg_list = [seg[0] for seg in seg_cnt]
logger.info(f"Ordered seg_list: {seg_list}")

# COMMAND ----------

dbutils.notebook.exit(str({"seg_list": seg_list}))

# COMMAND ----------



# COMMAND ----------



# COMMAND ----------

# MAGIC %md # dev

# COMMAND ----------

# config_bd = config["build_dataset"]

# COMMAND ----------

# persist_utils.get_table_name(
#             factory_database=config_bd.etl_data_tbl.factory_database,
#             lab_database=config.dev_database,
#             table_prefix=config_bd.etl_data_tbl.prefix,
#             sensitivity=config_bd.etl_data_tbl.sensitivity,
#         )

# COMMAND ----------

# %sql select * from loyalty_azlab_prod.headroom_etl_data_np_p_tbl

# COMMAND ----------



# COMMAND ----------

# import inspect
# lines = inspect.getsource(TransactionsManager.get)
# print(lines)

# COMMAND ----------


