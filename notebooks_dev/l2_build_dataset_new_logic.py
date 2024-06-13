# Databricks notebook source
# MAGIC %run ../notebooks/bootstrap

# COMMAND ----------

from ast import literal_eval
from datetime import datetime, timedelta

import offerallocationv2.utils.persist_utils as persist_utils
import seaborn as sns
from dtaml.logging import get_logger
from pyspark.sql import Window as W
from pyspark.sql import functions as F
from pyspark.sql import types as T

from customer_headroom.etl.build_dataset import TransactionsManager

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

# campaign = 20230807

logger.info(
    f"""
config_dates: {config_dates}
campaign: {campaign}
last_registration_date: {last_registration_date}
"""
)

# COMMAND ----------

# Get segmentations to use

logger.info("Begin building dataset")
config_bd = config["build_dataset"]
config_use = config["use_segments"]

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

# COMMAND ----------

# Initialize trx_manager class
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
)

# COMMAND ----------


# COMMAND ----------

# get() broken down:
# upto get transaction metrics


trx_line = trx_line_df
lu_article = articles_df
cust_seg = segmentations_tbl

trx_line = (
    trx_manager._add_date(trx_line)
    .filter(F.col("date") <= trx_manager.etl_date)
    .filter(F.col("date") >= trx_manager.lookback_date)
    .filter(F.col("PURCHASE_CHANNEL").isin(trx_manager.channels))
    .filter(F.col("l1_id").isin(list(trx_manager.l1_ids)))
    .filter(trx_manager.get_common_filters())
)

if trx_manager.christmas_remove_range is not None:
    trx_line = trx_manager.remove_christmas_transactions(
        trx_line, christmas_range=trx_manager.christmas_remove_range
    )

# Remove items from transaction list, e.g. BWS items
trx_line = trx_manager.remove_items(trx_line)

if cust_seg is not None:
    # Only keep customers in segmentations
    trx_line = trx_line.join(
        cust_seg.select(trx_manager.user_key).distinct(), on=trx_manager.user_key
    )

cust_lx_trx = trx_manager.get_customer_transactions(trx_line, lu_article)

# Add a time window column to groupby
if trx_manager.time_window_length is not None:
    time_window_ind_df = trx_manager.add_time_window_ind(cust_lx_trx=cust_lx_trx)
    cust_lx_trx = cust_lx_trx.join(time_window_ind_df, on="date", how="left")

# COMMAND ----------

"""cust_lx_trx.write.parquet("/mnt/centralds/offerallocation/headroom/analysis/231028/filter_cust_lx_trans", mode = "overwrite")
"""
# filter_cust_lx_trans = spark.read.parquet("/mnt/centralds/offerallocation/headroom/analysis/231028/filter_cust_lx_trans", mode = "overwrite")
# filter_cust_lx_trans.display()

# COMMAND ----------

# get transaction metrics broken down
customer_lx_transactions = cust_lx_trx
customer_lx_transactions.cache()

# COMMAND ----------

customer_lx_transactions.display()

# COMMAND ----------

# Find number of transactions per customer per l2 category
customer_lx_trans_grouped = (
    customer_lx_transactions.filter(F.col(trx_manager.user_key).isNotNull())
    .groupby([trx_manager.user_key, f"{trx_manager.lx}_id"])
    # .groupby([self.user_key])
    #                                      .pivot(f"{self.lx}_id")
    .agg(
        F.count(f"{trx_manager.lx}_name")
        .cast(T.IntegerType())
        .alias("number_of_transactions"),
        F.sum("sales_amt").cast(T.DoubleType()).alias("total_spend"),
        F.count("article_id").cast(T.IntegerType()).alias("items"),
        F.countDistinct("basket_id").cast(T.IntegerType()).alias("visits"),
        (F.sum("sales_amt") / F.count("article_id"))
        .cast(T.DoubleType())
        .alias("spend_per_item"),
    )
)

# COMMAND ----------

customer_lx_trans_grouped.display()

# COMMAND ----------

percentile_spend_time_window = (
    customer_lx_transactions.select("cust_id", "time_window_ind", "sales_amt")
    # find the spend in time window
    .groupby("cust_id", "time_window_ind")
    .agg(F.sum("sales_amt").cast(T.DoubleType()).alias("total_spend_time_window"))
    # find the weeky max basket amount
    # .groupby("cust_id", "time_window_ind")
    # .agg(F.max("total_spend_basket").cast(T.DoubleType()).alias("time_window_max_spend_basket"))
    .groupby("cust_id")
    .agg(*trx_manager.get_expr_agg("total_spend_time_window"))
    .select("cust_id", "85percentile_total_spend_time_window")
)

# COMMAND ----------

"""
this should come between customer_lx_trans_grouped and customer_overall_count -> replacing (percentile_spend_time_window,time_window_ind_id,customer_lx_time_window_spend) -> should be joined directly to customer_lx_trans_grouped
"""
df = (
    customer_lx_transactions.select(
        "cust_id", f"{trx_manager.lx}_id", "time_window_ind", "sales_amt"
    )
    # find the spend in time window for each lx_id
    .groupby("cust_id", f"{trx_manager.lx}_id", "time_window_ind")
    .agg(F.sum("sales_amt").cast(T.DoubleType()).alias("total_spend_time_window"))
    # get 85.00 percentile spend in time window for each lx_id
    .groupby("cust_id", f"{trx_manager.lx}_id")
    .agg(*trx_manager.get_expr_agg("total_spend_time_window"))
    .select("cust_id", f"{trx_manager.lx}_id", "85percentile_total_spend_time_window")
    .withColumn(
        f"{trx_manager.lx}_id_total_time_window_spend",
        F.col("85percentile_total_spend_time_window"),
    )
)

# COMMAND ----------

time_window_ind_id = (
    customer_lx_transactions.select("cust_id", "time_window_ind", "sales_amt")
    # find the basket amount
    .groupby("cust_id", "time_window_ind")
    .agg(F.sum("sales_amt").cast(T.DoubleType()).alias("total_spend_time_window"))
    # find the weeky max basket amount
    # .groupby("cust_id", "WEEK_ID")
    # .withColumn('time_window_max_spend_basket', F.max("total_spend_basket").over(W.partitionBy("cust_id", "time_window_ind") ))
    # .where(F.col("total_spend_basket") == F.col("time_window_max_spend_basket"))
    # .withColumn("percentile", F.lit(34.94))
    .join(percentile_spend_time_window, how="left", on="cust_id")
    .where(
        F.col("total_spend_time_window")
        >= F.col("85percentile_total_spend_time_window")
    )
    # .orderBy("time_window_max_spend_basket")
    .withColumn(
        "row",
        F.row_number().over(
            W.partitionBy("cust_id").orderBy(F.col("total_spend_time_window"))
        ),
    )
    .filter(F.col("row") == 1)
)

time_window_ind_id.cache()


# COMMAND ----------

# find the l2 id spend for the given time window
customer_lx_time_window_spend = (
    customer_lx_transactions.join(
        time_window_ind_id.select("cust_id", "time_window_ind"),
        how="inner",
        on=["cust_id", "time_window_ind"],
    )
    .groupby("cust_id", f"{trx_manager.lx}_id")
    .agg(
        F.sum("sales_amt")
        .cast(T.DoubleType())
        .alias(f"{trx_manager.lx}_id_total_time_window_spend")
    )
)

# COMMAND ----------

customer_overall_count = (
    customer_lx_transactions.filter(F.col(trx_manager.user_key).isNotNull())
    .groupby(trx_manager.user_key, f"{trx_manager.lx}_id")
    .agg(
        F.countDistinct("basket_id").cast(T.IntegerType()).alias("count_user_basket"),
        F.countDistinct("time_window_ind")
        .cast(T.IntegerType())
        .alias("count_user_time_window"),
    )
)

# COMMAND ----------

customer_overall_count.display()

# COMMAND ----------

customer_lx_trans_grouped_all = customer_lx_trans_grouped.join(
    customer_lx_time_window_spend, on=[trx_manager.user_key, f"{trx_manager.lx}_id"]
).join(customer_overall_count, on=[trx_manager.user_key])

# COMMAND ----------

df_all = customer_lx_trans_grouped.join(
    df, on=[trx_manager.user_key, f"{trx_manager.lx}_id"]
).join(customer_overall_count, on=[trx_manager.user_key, f"{trx_manager.lx}_id"])

# COMMAND ----------

cust_lx_trx_metrics = df_all
trx_timespan = trx_manager.add_timespan_spend(cust_lx_trx)
cust_lx_trx_metrics = cust_lx_trx_metrics.join(
    trx_timespan, on=trx_manager.user_key, how="left"
).fillna(0)

# COMMAND ----------

cust_lx_trx_metrics = cust_lx_trx_metrics.join(
    cust_seg.dropDuplicates(subset=[trx_manager.user_key]), on=trx_manager.user_key
)

# COMMAND ----------

cust_lx_trx_metrics.display()

# COMMAND ----------


# COMMAND ----------

cust_lx_trx_metrics.select(
    "cust_id",
    "l2_id",
    "number_of_transactions",
    "count_user_basket",
    "count_user_time_window",
    "l2_id_total_time_window_spend",
).display()

# COMMAND ----------

# MAGIC %md
# MAGIC trying L3

# COMMAND ----------

trx_manager = TransactionsManager(
    etl_date=get_date(config_bd["etl_date"]),
    lookback_days=config_bd["lookback_days"],
    l1_ids=config_bd["l1_ids"],
    lx="l3",
    lx_ids=config_bd["lx_ids"],  # getting all the products in this l2 id
    user_key=config_bd["user_id"],
    window_days=config_bd["window_days"],
    time_window_length=config_bd["time_window_days"],
    exclude_items=literal_eval(config["exclude_items"]),
)

# COMMAND ----------

trx_line = trx_line_df
lu_article = articles_df
cust_seg = segmentations_tbl

trx_line = (
    trx_manager._add_date(trx_line)
    .filter(F.col("date") <= trx_manager.etl_date)
    .filter(F.col("date") >= trx_manager.lookback_date)
    .filter(F.col("PURCHASE_CHANNEL").isin(trx_manager.channels))
    .filter(F.col("l1_id").isin(list(trx_manager.l1_ids)))
    .filter(trx_manager.get_common_filters())
)

if trx_manager.christmas_remove_range is not None:
    trx_line = trx_manager.remove_christmas_transactions(
        trx_line, christmas_range=trx_manager.christmas_remove_range
    )

# Remove items from transaction list, e.g. BWS items
trx_line = trx_manager.remove_items(trx_line)

if cust_seg is not None:
    # Only keep customers in segmentations
    trx_line = trx_line.join(
        cust_seg.select(trx_manager.user_key).distinct(), on=trx_manager.user_key
    )

# COMMAND ----------

trx_line.display()

# COMMAND ----------

# cust_lx_trx = trx_manager.get_customer_transactions(trx_line, lu_article)

customer_transactions = trx_line.select(
    trx_manager.user_key,
    "cust_age",
    "cust_gender",
    "article_id",
    "basket_id",
    "sales_amt",
    "date",
).filter(F.col("SALES_AMT") > 0.5)
"""
lx_all = (lu_article
                  .filter(lu_article[f"{self.lx}_id"].isin(list(self.lx_ids)))
                  .select(["article_id"] +
                          [f"l{i}_id" for i in range(1, 7)] +
                          [f"l{i}_name" for i in range(1, 7)]
                          )
                  )
"""
lx_all = lu_article.filter(F.col("l1_id") == "FD").select(
    "article_id", "l2_id", "l2_name", "l3_id", "l3_name"
)
customer_lx_transactions = customer_transactions.join(lx_all, ["article_id"])

# COMMAND ----------

customer_lx_transactions.display()

# COMMAND ----------

cust_lx_trx = customer_lx_transactions

# Add a time window column to groupby
if trx_manager.time_window_length is not None:
    time_window_ind_df = trx_manager.add_time_window_ind(cust_lx_trx=cust_lx_trx)
    cust_lx_trx = cust_lx_trx.join(time_window_ind_df, on="date", how="left")

customer_lx_transactions = cust_lx_trx
customer_lx_transactions.cache()


# COMMAND ----------

# Find number of transactions per customer per l2 category
customer_lx_trans_grouped = (
    customer_lx_transactions.filter(F.col(trx_manager.user_key).isNotNull())
    .groupby([trx_manager.user_key, f"{trx_manager.lx}_id"])
    # .groupby([self.user_key])
    #                                      .pivot(f"{self.lx}_id")
    .agg(
        F.count(f"{trx_manager.lx}_name")
        .cast(T.IntegerType())
        .alias("number_of_transactions"),
        F.sum("sales_amt").cast(T.DoubleType()).alias("total_spend"),
        F.count("article_id").cast(T.IntegerType()).alias("items"),
        F.countDistinct("basket_id").cast(T.IntegerType()).alias("visits"),
        (F.sum("sales_amt") / F.count("article_id"))
        .cast(T.DoubleType())
        .alias("spend_per_item"),
    )
)

# COMMAND ----------


df = (
    customer_lx_transactions.select(
        "cust_id", f"{trx_manager.lx}_id", "time_window_ind", "sales_amt"
    )
    # find the spend in time window for each lx_id
    .groupby("cust_id", f"{trx_manager.lx}_id", "time_window_ind")
    .agg(F.sum("sales_amt").cast(T.DoubleType()).alias("total_spend_time_window"))
    # get 85.00 percentile spend in time window for each lx_id
    .groupby("cust_id", f"{trx_manager.lx}_id")
    .agg(*trx_manager.get_expr_agg("total_spend_time_window"))
    .select("cust_id", f"{trx_manager.lx}_id", "85percentile_total_spend_time_window")
    .withColumn(
        f"{trx_manager.lx}_id_total_time_window_spend",
        F.col("85percentile_total_spend_time_window"),
    )
)

customer_overall_count = (
    customer_lx_transactions.filter(F.col(trx_manager.user_key).isNotNull())
    .groupby(trx_manager.user_key)
    .agg(
        F.countDistinct("basket_id").cast(T.IntegerType()).alias("count_user_basket"),
        F.countDistinct("time_window_ind")
        .cast(T.IntegerType())
        .alias("count_user_time_window"),
    )
)

df_all = customer_lx_trans_grouped.join(
    df, on=[trx_manager.user_key, f"{trx_manager.lx}_id"]
).join(customer_overall_count, on=[trx_manager.user_key])
cust_lx_trx_metrics = df_all
trx_timespan = trx_manager.add_timespan_spend(cust_lx_trx)
cust_lx_trx_metrics = cust_lx_trx_metrics.join(
    trx_timespan, on=trx_manager.user_key, how="left"
).fillna(0)

cust_lx_trx_metrics = cust_lx_trx_metrics.join(
    cust_seg.dropDuplicates(subset=[trx_manager.user_key]), on=trx_manager.user_key
)

# COMMAND ----------

cust_lx_trx_metrics.display()

# COMMAND ----------

cust_lx_trx_metrics.filter(F.col("cust_id") == -1000822876767570416).display()

# COMMAND ----------


# COMMAND ----------
