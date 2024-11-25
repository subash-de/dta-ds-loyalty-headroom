# Databricks notebook source
# MAGIC %run ../setup/bootstrap

# COMMAND ----------

from ast import literal_eval
from datetime import datetime, timedelta

import seaborn as sns
from dtaml.logging import get_logger
from dtaml.utils.table import factory_table

import customer_headroom.utils.persist_utils as persist_utils
from customer_headroom.etl.build_dataset import (
    TransactionsManager, 
    TransactionsManagerFixedStretch, 
    TransactionsManagerOneUnitStretch,
)
from customer_headroom.etl.etl_utils import (
    find_all_segments,
    get_campaign,
    get_count,
    get_date,
)

from pyspark.sql import functions as F, DataFrame, Column, Window as W, types as T


sns.set(style="whitegrid")
logger = get_logger("customer-headroom")

# COMMAND ----------

config.dates.etl_date

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

# Step 3: Get seg_list: List of dictionaries
#   where each dictionary has the following properties:
#       - campaign id
#       - experian_hh_composition
#       - segmentation id (of each experian_hh_composition)

config_use = config["use_segments"]
segmentations_tbl_name = factory_table(
    table_prefix=config_use.segmentations_tbl.prefix, sensitivity=config.sensitivity
)
segmentations_tbl = persist_utils.read_table(
    table_name=segmentations_tbl_name, where=f"campaign={campaign}"
)
if config_use["all"]:
    logger.info("Use all Segmentations")
    seg_list = find_all_segments(segmentations_tbl, config_use["partitionByList"])
else:
    seg_list = config_use["seg_list"]
logger.info(f"Segmentations: {seg_list}")

# COMMAND ----------

build_dataset = widgets.get("build_dataset", "True")

# COMMAND ----------

config_bd = config["build_dataset"]
print(config_bd)

# COMMAND ----------

# Load factory tables
articles_df = persist_utils.read_table(
  table_name = config.factory_tbl_lu_article
)
trx_line_df = persist_utils.read_table(
  table_name = config.factory_tbl_all_transaction_line
)
sparks_account_df = persist_utils.read_table(
  table_name = config.factory_tbl_sparks_account
)
segtco_history_df = persist_utils.read_table(
  table_name = config.factory_tbl_segtco_history
)
# COMMAND ----------

if build_dataset == "True":
    # Step 4: Build Dataset
    logger.info("Begin building dataset")
    config_bd = config["build_dataset"]

    # Load Segmentation Dataset
    # TODO: Replace with customer cluster work to reduce data sizes to appropiate groups.
    # TODO: Possibl build data for just 1 segment at a time?
    partitionByList = config_use.segmentations_tbl.partitionByList 
    ###################################
    # # build training data
    # trx_manager = TransactionsManager(
    #     etl_date=get_date(config.dates.etl_date),
    #     lookback_days=config.dates.lookback_days,
    #     l1_ids=config_bd["l1_ids"],
    #     category_level=config["category_level"],
    #     l2_ids=config_bd["l2_ids"],
    #     lx=config_bd["lx"],
    #     lx_ids=config_bd["lx_ids"],  # getting all the products in this l2 id
    #     user_key=config_bd["user_id"],
    #     window_days=config_bd["window_days"],
    #     time_window_length=config_bd["time_window_days"],
    #     exclude_items=literal_eval(config["exclude_items"]),
    #     aggregation_level=config_bd["aggregation_level"],
    # )

    # all_data = trx_manager.get(trx_line_df, articles_df, cust_seg=segmentations_tbl)

    # etl_data_tbl_name = persist_utils.create_beam_table(
    #     table_prefix=config_bd.etl_data_tbl.prefix,
    #     lab_database=config.lab_database,
    #     factory_database=config.factory_database,
    #     sensitivity=config.sensitivity,
    #     schema=all_data,
    #     partition_by=config_bd.etl_data_tbl.partitionByList,
    #     overwrite_table=False,
    #     assert_equality=False,
    #     add_load_timestamp=True,
    # )
    # logger.info(f"""etl_data_tbl_name: {etl_data_tbl_name}""")

    # persist_utils.insert_df_into_table(
    #     target_tbl_name=etl_data_tbl_name,
    #     insert_df=all_data,
    #     add_columns=True,
    #     insert_append=True,
    #     delete_where=f"campaign={campaign}",
    # )
    ###################################
    # Step 4: Build Dataset
    logger.info("Begin building dataset")
    config_bd = config["build_dataset"]

    # Load Segmentation Dataset
    # TODO: Replace with customer cluster work to reduce data sizes to appropiate groups.
    # TODO: Possibl build data for just 1 segment at a time?
    partitionByList = config_use.segmentations_tbl.partitionByList

    segmentations_tbl_name = factory_table(
        table_prefix=config_bd.segmentations_tbl.prefix, sensitivity=config.sensitivity
    )

    logger.info(f"""segmentations_tbl_name: {segmentations_tbl_name}""")

    segmentations_tbl = persist_utils.read_table(
        table_name=segmentations_tbl_name, where=f"campaign= {campaign}"
    ).select([config_bd["user_id"]] + partitionByList)

    # build training data
    trx_manager = TransactionsManager(
        etl_date=get_date(config.dates.etl_date),
        lookback_days=config.dates.lookback_days,
        l1_ids=config_bd["l1_ids"],
        category_level=config["category_level"],
        l2_ids=config_bd["l2_ids"],
        lx=config_bd["lx"],
        lx_ids=config_bd["lx_ids"], # getting all the products in this l2 id
        user_key=config_bd["user_id"],
        window_days=config_bd["window_days"],
        time_window_length=config_bd["time_window_days"],
        exclude_items=literal_eval(config["exclude_items"]),
        aggregation_level=config_bd["aggregation_level"],
    )

    all_data = trx_manager.get(trx_line_df, articles_df, cust_seg=segmentations_tbl)

    etl_data_tbl_name = persist_utils.create_beam_table(
        table_prefix=config_bd.etl_data_tbl.prefix,
        lab_database=config.lab_database,
        factory_database=config.factory_database,
        sensitivity=config.sensitivity,
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
    # TODO: delete all mentions of validationmanager

    etl_data_tbl = persist_utils.read_table(
        table_name=etl_data_tbl_name, where=f"campaign={campaign}"
    )
    sparks = persist_utils.read_table(
        table_name = config.factory_tbl_sparks_account
        ).select("account_id", "uk_digital_id",  "cust_id")
    sparks = (sparks.withColumn("row",F.row_number().over(W.partitionBy("cust_id").orderBy("account_id") )).filter(F.col("row") == 1).drop("row"))

    cust_id_link = (
        etl_data_tbl
        .select("cust_id")
        .distinct()
        .join(sparks, how = 'left', on = 'cust_id')
        .withColumn('campaign', F.lit(campaign))
    )
    cust_id_link_tbl_name = persist_utils.create_beam_table(
        table_prefix=config_bd.headroom_cust_id_link_tbl.prefix,
        lab_database=config.lab_database,
        factory_database=config.factory_database,
        sensitivity=config.sensitivity,
        schema=cust_id_link,
        partition_by=config_bd.headroom_cust_id_link_tbl.partitionByList,
        overwrite_table=False,
        assert_equality=False,
        add_load_timestamp=True,
    )
    logger.info(f"""cust_id_link_tbl_name: {cust_id_link_tbl_name}""")

    persist_utils.insert_df_into_table(
        target_tbl_name=cust_id_link_tbl_name,
        insert_df=cust_id_link,
        add_columns=True,
        insert_append=True,
        delete_where=f"campaign={campaign}",
    )


    # Step 5: Get ordered segmentation list
    seg_cnt = []
    for seg in seg_list:
        config_bd = config["build_dataset"]
        get_seg_cnt = get_count(
                seg,
                config=config,
            )
        # Record the segment in the seg_list only if the segment count is larger than zero
        if get_seg_cnt[1] > 0:
            seg_cnt.append(get_seg_cnt)
        else:
            logger.info(f"Segment {seg} has no data, skipping")

    seg_cnt.sort(key=lambda i: i[1], reverse=True)

    seg_list = [seg[0] for seg in seg_cnt]
    logger.info(f"Ordered seg_list: {seg_list}")


# COMMAND ----------

etl_data_tbl.select(f'{config_bd["lx"]}_id').distinct().display()

# COMMAND ----------

print(seg_cnt)

# COMMAND ----------

# seg_list has
# dbutils.notebook.exit(str({"seg_list": seg_list}))

# COMMAND ----------

# Building data for baseline + fixed stretch approach
config_bd = config["build_dataset"]
config_sim = config["baseline_stretch_simulations"]
config_sim["rolling_window_col"] = f"rolling_{config_sim['rolling_window']}_week_sales"

# COMMAND ----------

trx_manager_fixed_stretch = TransactionsManagerFixedStretch(
    baseline_percentiles=config_sim['baseline_percentiles'],
    etl_date=get_date(config.dates.etl_date),
    lookback_days=config.dates.lookback_days,
    grouping_columns=config_sim["grouping_columns"],
    rolling_window=config_sim["rolling_window"],
    rolling_window_col = config_sim["rolling_window_col"],
    l1_ids=config_bd["l1_ids"],
    category_level=config["category_level"],
    l2_ids=config_bd["l2_ids"],
    lx=config_bd["lx"],
    lx_ids=config_bd["lx_ids"],  # getting all the products in this l2 id
    user_key=config_bd["user_id"],
    exclude_items=literal_eval(config["exclude_items"]),
)

weekly_data = trx_manager_fixed_stretch.get(trx_line_df, articles_df)

# COMMAND ----------

# If the prediction is on category level, we only keep the relevant categories
if config["category_level"] == True:
  weekly_data = weekly_data.filter(F.col(f'{config_bd["lx"]}_id').isin(list(config_bd["lx_ids"].keys())))

# COMMAND ----------

fixed_stretch_etl_data_tbl_name= persist_utils.create_beam_table(
    table_prefix=config_bd.fixed_stretch_etl_data_tbl.prefix,
    lab_database=config.lab_database,
    factory_database=config.factory_database,
    sensitivity=config.sensitivity,
    schema=weekly_data,
    partition_by=config_bd.fixed_stretch_etl_data_tbl.partitionByList,
    overwrite_table=True,
    assert_equality=False,
    add_load_timestamp=True,
)
logger.info(f"""fixed_stretch_etl_data_tbl_name: {fixed_stretch_etl_data_tbl_name}""")

persist_utils.insert_df_into_table(
    target_tbl_name=fixed_stretch_etl_data_tbl_name,
    insert_df=weekly_data,
    insert_append=True,
    add_columns=True,
)

# COMMAND ----------

fixed_stretch_etl_data_tbl_name = persist_utils.get_table_name(
  factory_database=config.factory_database,
  lab_database=config.lab_database,
  table_prefix=config_bd.fixed_stretch_etl_data_tbl.prefix,
  sensitivity=config.sensitivity
)

logger.info(f"""fixed_stretch_etl_data_tbl_name: {fixed_stretch_etl_data_tbl_name}""")

fixed_stretch_etl_data_tbl = persist_utils.read_table(
    table_name=fixed_stretch_etl_data_tbl_name
)

# COMMAND ----------

fixed_stretch_etl_data_tbl.display()

# COMMAND ----------

# Building data for one article unit stretch approach
trx_manager_one_unit_stretch = TransactionsManagerOneUnitStretch(
        etl_date=get_date(config.dates.etl_date),
        lookback_days=config.dates.lookback_days,
        grouping_columns=config_sim["grouping_columns"],
        rolling_window=config_sim["rolling_window"],
        rolling_window_col = config_sim["rolling_window_col"],
        l1_ids=config_bd["l1_ids"],
        category_level=config["category_level"],
        l2_ids=config_bd["l2_ids"],
        lx=config_bd["lx"],
        lx_ids=config_bd["lx_ids"],  # getting all the products in this l2 id
        user_key=config_bd["user_id"],
        exclude_items=literal_eval(config["exclude_items"]),
    )

customer_one_additional_unit_price_stretch = trx_manager_one_unit_stretch.get(trx_line_df, 
                                               articles_df, 
                                               config_sim["percentile_for_one_additional_unit_price"], 
                                               config_sim["percentile_for_customer_one_additional_unit_price"],
                                               config_sim["article_threshold_for_fallback"])

# COMMAND ----------

# If the prediction is on category level, we only keep the relevant categories
if config["category_level"] == True:
  customer_one_additional_unit_price_stretch = customer_one_additional_unit_price_stretch.filter(
    customer_one_additional_unit_price_stretch[f'{config_bd["lx"]}_id'].isin(list(config_bd["lx_ids"].keys()))
  )

# COMMAND ----------

one_article_unit_stretch_etl_data_tbl_name= persist_utils.create_beam_table(
    table_prefix=config_bd.one_article_unit_stretch_etl_data_tbl.prefix,
    lab_database=config.lab_database,
    factory_database=config.factory_database,
    sensitivity=config.sensitivity,
    schema=customer_one_additional_unit_price_stretch,
    partition_by=config_bd.one_article_unit_stretch_etl_data_tbl.partitionByList,
    overwrite_table=True,
    assert_equality=False,
    add_load_timestamp=True,
)
logger.info(f"""one_article_unit_stretch_etl_data_tbl_name: {one_article_unit_stretch_etl_data_tbl_name}""")

persist_utils.insert_df_into_table(
    target_tbl_name=one_article_unit_stretch_etl_data_tbl_name,
    insert_df=customer_one_additional_unit_price_stretch,
    insert_append=True,
    add_columns=True,
)

# COMMAND ----------

one_article_unit_stretch_etl_data_tbl_name = persist_utils.get_table_name(
   factory_database=config.factory_database,
    lab_database=config.lab_database,
  table_prefix=config_bd.one_article_unit_stretch_etl_data_tbl.prefix,
  sensitivity=config.sensitivity
)

logger.info(f"""one_article_unit_stretch_etl_data_tbl_name: {one_article_unit_stretch_etl_data_tbl_name}""")

one_article_unit_stretch_etl_data_tbl = persist_utils.read_table(
    table_name=one_article_unit_stretch_etl_data_tbl_name
)

# COMMAND ----------

one_article_unit_stretch_etl_data_tbl.display()

# COMMAND ----------

dbutils.notebook.exit(str({"seg_list": seg_list}))
