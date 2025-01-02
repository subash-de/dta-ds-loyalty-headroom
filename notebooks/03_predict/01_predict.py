# Databricks notebook source
# MAGIC %run ../bootstrap

# COMMAND ----------

dbutils.widgets.text("seg_list", "[]", "")

# COMMAND ----------

from datetime import datetime, timedelta

import seaborn as sns
from dtaml.logging import get_logger
from pyspark.sql import functions as F

import customer_headroom.utils.persist_utils as persist_utils
from customer_headroom.modelling.predict import Predictor,PredictorFixedStretch

sns.set(style="whitegrid")
logger = get_logger("customer-headroom")

# COMMAND ----------

seg_list = eval(dbutils.widgets.get("seg_list"))
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


logger.info("Begin Predictions")
config_pd = config["predict"]
config_sg = config['segmentation']
partitionByList = config_pd["partitionByList"]

# get list of pred items depending on lx id and whether the prediction is on category level or not
lu_article = persist_utils.read_table(
  table_name = config.factory_tbl_lu_article
)

pred_items = (
    lu_article.filter(F.col("l2_id").isin(config_pd["l2_ids"]))
    .select(f'{config_pd["pred_key"]}_id')
    .distinct()
    .toPandas()
)
if config["category_level"]:
    offer_pred_ids = list(config_pd['pred_items'].keys())
    # If predicting on the category level, predict items are all existing ids and synthetic ones
    list_pred_items = list(pred_items[f'{config_pd["pred_key"]}_id']) + offer_pred_ids
else:
    # If predicting on the full basket level, predict items are only all existing ids
    list_pred_items = list(pred_items[f'{config_pd["pred_key"]}_id'])

etl_data_tbl_name = persist_utils.get_table_name(
    factory_database=config.factory_database,
    lab_database=config.lab_database,
    table_prefix=config_pd.etl_data_tbl.prefix,
    sensitivity=config.sensitivity,
)


# COMMAND ----------

list_pred_items

# COMMAND ----------

first_segment = True
for seg in seg_list:
    seg_ext = [f"({k}='{seg[k]}')" for k in partitionByList]
    ext_str = "_".join([str(seg[k]) for k in partitionByList])

    data = persist_utils.read_table(
        table_name=etl_data_tbl_name, where=" and ".join(seg_ext)
    )

    rec_name = (config_pd.rec_name + "_{ext}" + "{model_prefix}").format(
        ext=ext_str, 
        model_prefix=config_pd["model_prefix"],
    )
    logger.info(f"{seg}: Read Recommender name={rec_name}")
    rec_algo = persist_utils.get_latest_version(model_name=rec_name)

    data_processor_name = (config_pd.data_processor_name + "_{ext}" + "{model_prefix}").format(
        ext=ext_str, 
        model_prefix=config_pd["model_prefix"],
    )
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
        .withColumn("l1_id", F.lit(config_sg['l1_id']))
        .withColumn("category_level", F.lit(config['category_level']))
        .drop("load_timestamp")
        .withColumn("experian_hh_composition", F.lit(seg["experian_hh_composition"]))
        .withColumn("segmentation", F.lit(seg["segmentation"]))
    )  # ----------------------------------------------
    # If the prediction is on category level, we only need to save the relevant categories
    if config["category_level"]:
        predictions = predictions.filter(predictions[f'{config_pd["pred_key"]}_id'].isin(offer_pred_ids))

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

    # TODOL change this part to writing to a staging table first
    if first_segment:
        # if this is the first segment, delete all partitions related to the current campaign
        delete_where_statement = f"campaign={campaign} and l1_id = '{config_sg['l1_id']}' and category_level={config['category_level']}"
        first_segment = False
    else:
        delete_where_statement = None

    persist_utils.insert_df_into_table(
        target_tbl_name=prediction_tbl_name,
        insert_df=predictions,
        delete_where=delete_where_statement,
    )

    predictions_read = persist_utils.read_table(
        table_name=prediction_tbl_name, where=" and ".join(seg_ext)
    )
    logger.info(
        f"predictions {seg} | row count: {predictions_read.count()}; column count: {len(predictions_read.columns)}"
    )

# COMMAND ----------


prediction_tbl_name = persist_utils.get_table_name(
    factory_database=config.factory_database,
    lab_database=config.lab_database,
    table_prefix=config_pd.prediction_tbl.prefix,
    sensitivity=config.sensitivity,
)

logger.info(f"""prediction_tbl_name: {prediction_tbl_name}""")

prediction_tbl = persist_utils.read_table(
    table_name=prediction_tbl_name, where=f"campaign={campaign} and l1_id = '{config_sg['l1_id']}' and category_level={config['category_level']}"
)

# COMMAND ----------

prediction_tbl.display()

# COMMAND ----------

# Fixed stretch predict
if config["fixed_stretch"]:
    fixed_stretch_etl_data_tbl_name = persist_utils.get_table_name(
        factory_database=config.factory_database,
        lab_database=config.lab_database,
        table_prefix=config_pd.fixed_stretch_etl_data_tbl.prefix,
        sensitivity=config.sensitivity,
    )

    logger.info(f"""fixed_stretch_etl_data_tbl_name: {fixed_stretch_etl_data_tbl_name}""")

    fixed_stretch_etl_tbl = persist_utils.read_table(
        table_name=fixed_stretch_etl_data_tbl_name,
        where=f"campaign={campaign} and l1_id = '{config_sg['l1_id']}' and category_level={config['category_level']}"
    )


    config_sim = config["baseline_stretch_simulations"]
    config_sim["rolling_window_col"] = f"rolling_{config_sim['rolling_window']}_week_sales"

    fixed_stretch_predicition_manager = PredictorFixedStretch(
        grouping_columns = config_sim['grouping_columns'],
        rolling_window_col = config_sim["rolling_window_col"],
        baseline_percentiles = config_sim['baseline_percentiles'],
        stretch_amounts = config_sim['stretch_amounts'],                               
    )
    baseline_per_customer = fixed_stretch_predicition_manager.get(fixed_stretch_etl_tbl)
    baseline_per_customer.display()
    baseline_per_customer = baseline_per_customer.withColumn('campaign', F.lit(campaign))
    baseline_per_customer = baseline_per_customer.withColumn('l1_id', F.lit(config_sg['l1_id']))
    baseline_per_customer = baseline_per_customer.withColumn('category_level', F.lit(config['category_level']))

    fixed_stretch_tbl_name= persist_utils.create_beam_table(
        table_prefix=config_sim.fixed_stretch_tbl.prefix,
        lab_database=config.lab_database,
        factory_database=config.factory_database,
        sensitivity=config.sensitivity,
        schema=baseline_per_customer,
        partition_by=config_sim.fixed_stretch_tbl.partitionByList,
        overwrite_table=False,
        assert_equality=False,
        add_load_timestamp=True,
    )
    logger.info(f"""fixed_stretch_tbl_name: {fixed_stretch_tbl_name}""")

    persist_utils.insert_df_into_table(
        target_tbl_name=fixed_stretch_tbl_name,
        insert_df=baseline_per_customer,
        insert_append=True,
        add_columns=True,
        delete_where=f"campaign={campaign} and l1_id = '{config_sg['l1_id']}' and category_level={config['category_level']}"
    )

    fixed_stretch_tbl_name = persist_utils.get_table_name(
        factory_database=config.factory_database,
        lab_database=config.lab_database,
        table_prefix=config_sim.fixed_stretch_tbl.prefix,
        sensitivity=config.sensitivity,
    )

    logger.info(f"""fixed_stretch_tbl_name: {fixed_stretch_tbl_name}""")


    fixed_stretch_tbl = persist_utils.read_table(
        table_name=fixed_stretch_tbl_name,
        where=f"campaign={campaign} and l1_id = '{config_sg['l1_id']}' and category_level={config['category_level']}"
    )

# COMMAND ----------

# One article unit stretch predict
if config["one_article_unit_stretch"]:
    config_pd = config["predict"]
    one_article_unit_stretch_etl_data_tbl_name = persist_utils.get_table_name(
        factory_database=config.factory_database,
        lab_database=config.lab_database,
        table_prefix=config_pd.one_article_unit_stretch_etl_data_tbl.prefix,
        sensitivity=config.sensitivity,
    )

    logger.info(f"""one_article_unit_stretch_etl_data_tbl_name: {one_article_unit_stretch_etl_data_tbl_name}""")
    one_article_unit_stretch_etl_data_tbl = persist_utils.read_table(
        table_name=one_article_unit_stretch_etl_data_tbl_name,
        where=f"campaign={campaign} and l1_id = '{config_sg['l1_id']}' and category_level={config['category_level']}"
    )
    # Get one additional unit price for each category
    one_additional_unit_price_per_id = (one_article_unit_stretch_etl_data_tbl
                                        .groupby(f'{config_pd["pred_key"]}_id')
                                        .agg(F.max('one_additional_unit_price').alias('one_additional_unit_price'))
    )

    # Join the table for baseline
    one_article_unit_stretch_tbl = (fixed_stretch_tbl
                                    .select(["cust_id", f'{config_pd["pred_key"]}_id', "85th_percentile"])
                                    .join(
                                        one_article_unit_stretch_etl_data_tbl.select(
                                            ["cust_id", f'{config_pd["pred_key"]}_id', "customer_one_additional_unit_price_final", "avg_weekly_article_count"]),
                                        on=["cust_id", f'{config_pd["pred_key"]}_id'],
                                        how="left"
                                    ))

    # Join the table for the fallback article unit price
    one_article_unit_stretch_tbl = one_article_unit_stretch_tbl.join(one_additional_unit_price_per_id,
                                                                     on=[f'{config_pd["pred_key"]}_id'], how='left')


    # Fill in the fallback article unit price if the customized article unit price is null
    one_article_unit_stretch_tbl = (one_article_unit_stretch_tbl
                                    .withColumn("customer_one_additional_unit_price_stretch", 
                                                F.when(
                                                F.col("customer_one_additional_unit_price_final").isNull(), 
                                                F.col("one_additional_unit_price")
                                                ).otherwise(F.col("customer_one_additional_unit_price_final")))
    )

    # Double check all customers have the one additional unit price stretch
    assert one_article_unit_stretch_tbl.filter(one_article_unit_stretch_tbl["customer_one_additional_unit_price_stretch"].isNull()).count() == 0, "Not all customers have the one additional unit price stretch"

    # Add the one article unit price stretch to the baseline
    one_article_unit_stretch_tbl = (one_article_unit_stretch_tbl
                                    .withColumn("85_stretch_one_article_unit", 
                                                F.col("85th_percentile") + 
                                                F.col("customer_one_additional_unit_price_stretch"))
    )

    # Drop irrelevant columns
    one_article_unit_stretch_tbl = one_article_unit_stretch_tbl.drop("one_additional_unit_price", "customer_one_additional_unit_price_final")

    # Fill in the avg_weekly_article_count if null
    one_article_unit_stretch_tbl = one_article_unit_stretch_tbl.fillna(0, subset=["avg_weekly_article_count"])

    one_article_unit_stretch_tbl = one_article_unit_stretch_tbl.withColumn('campaign', F.lit(campaign))
    one_article_unit_stretch_tbl = one_article_unit_stretch_tbl.withColumn('l1_id', F.lit(config_sg['l1_id']))
    one_article_unit_stretch_tbl = one_article_unit_stretch_tbl.withColumn('category_level', F.lit(config['category_level']))

    one_article_unit_stretch_tbl_name= persist_utils.create_beam_table(
        table_prefix=config_sim.one_article_unit_stretch_tbl.prefix,
        lab_database=config.lab_database,
        factory_database=config.factory_database,
        sensitivity=config.sensitivity,
        schema=one_article_unit_stretch_tbl,
        partition_by=config_sim.one_article_unit_stretch_tbl.partitionByList,
        overwrite_table=False,
        assert_equality=False,
        add_load_timestamp=True,
    )
    logger.info(f"""one_article_unit_stretch_tbl_name: {one_article_unit_stretch_tbl_name}""")

    persist_utils.insert_df_into_table(
        target_tbl_name=one_article_unit_stretch_tbl_name,
        insert_df=one_article_unit_stretch_tbl,
        insert_append=True,
        add_columns=True,
        delete_where=f"campaign={campaign} and l1_id = '{config_sg['l1_id']}' and category_level={config['category_level']}"
    )

    one_article_unit_stretch_tbl_name = persist_utils.get_table_name(
        factory_database=config.factory_database,
        lab_database=config.lab_database,
        table_prefix=config_sim.one_article_unit_stretch_tbl.prefix,
        sensitivity=config.sensitivity
        )

    logger.info(f"""one_article_unit_stretch_tbl_name: {one_article_unit_stretch_tbl_name}""")

    one_article_unit_stretch_tbl = persist_utils.read_table(
        table_name=one_article_unit_stretch_tbl_name,
        where=f"campaign={campaign} and l1_id = '{config_sg['l1_id']}' and category_level={config['category_level']}"
    )

# COMMAND ----------

dbutils.notebook.exit(True)
