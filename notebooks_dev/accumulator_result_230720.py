# Databricks notebook source
# MAGIC %md "feature_col": "l2_id_total_time_window_spend"

# COMMAND ----------

# MAGIC %run ../notebooks/bootstrap

# COMMAND ----------

config

# COMMAND ----------

import pandas as pd
import plotly
import plotly.express as px
import seaborn as sns
from offerallocationv2.utils import tmo_utils
from pyspark.ml.feature import Bucketizer
from pyspark.sql import Column, DataFrame
from pyspark.sql import Window as W
from pyspark.sql import functions as F
from pyspark.sql import types as T

# COMMAND ----------

# MAGIC %md # Data

# COMMAND ----------

# MAGIC %md ## TCOL segmentation

# COMMAND ----------

# MAGIC %md tcol segmentation, saving the file, as it takes a long time to load this data each time

# COMMAND ----------

# segtco_history_df = spark.table("customer_azbase_prod.segtco_history")
# segtco_history_ = tmo_utils.get_preceding_segtco_history(segtco_history_df, 20230720)
# segtco_history_.write.parquet("/mnt/centralds/offerallocation/headroom/analysis/230720/tcol_segmentation_230720")

# COMMAND ----------

segtco_history_ = spark.read.parquet(
    "/mnt/centralds/offerallocation/headroom/analysis/230720/tcol_segmentation_230720"
)
segtco_history_.count()

# COMMAND ----------

segtco_history_.display()

# COMMAND ----------

# MAGIC %md ## Accumulator 2023 07 20

# COMMAND ----------

# %sql select * from loyalty_azlab_prod.headroom_allocation_p_tbl
# %sql select campaign, count (distinct campaign) from loyalty_azlab_prod.headroom_allocation_p_tbl group by campaign

# COMMAND ----------

# MAGIC %sql select campaign, count (distinct campaign) from loyalty_azlab_prod.headroom_allocation_np_p_tbl group by campaign

# COMMAND ----------

# MAGIC %md  Saving the allocation result, because the allocation gets overwritten each time

# COMMAND ----------

# alloc = spark.sql("select * from loyalty_azlab_prod.headroom_allocation_np_p_tbl")
# # # # # alloc.groupby("campaign").count().show()
# alloc.write.parquet("/mnt/centralds/offerallocation/headroom/analysis/230720/allocation_accumulator", mode = "overwrite")


# COMMAND ----------

# alloc = spark.sql("select * from loyalty_azlab_prod.headroom_allocation_p_tbl")
# alloc.write.parquet("/mnt/centralds/offerallocation/headroom/analysis/230426/allocation")

alloc = spark.read.parquet(
    "/mnt/centralds/offerallocation/headroom/analysis/230720/allocation_accumulator"
)
alloc.count()

# COMMAND ----------

# distinct customers
alloc.select("cust_id").distinct().count()

# COMMAND ----------

alloc.display()


# COMMAND ----------

alloc.filter(
    (F.col("estimated_headroom") / F.col("estimated_spend") >= 0.0499)
    & (F.col("estimated_headroom") / F.col("estimated_spend") <= 0.0501)
).count()

# COMMAND ----------

alloc.filter(
    (F.col("estimated_headroom") / F.col("estimated_spend") >= 0.1999)
    & (F.col("estimated_headroom") / F.col("estimated_spend") <= 0.2001)
).count()

# COMMAND ----------


# COMMAND ----------

# MAGIC %md ## Spend and Save 2023 07 19

# COMMAND ----------

ss_result = spark.read.parquet(
    "/mnt/centralds/offerallocation/headroom/analysis/230719/allocation_ss"
)
ss_result.display()

# COMMAND ----------

(
    ss_result.groupBy("desc")
    .count()
    .withColumn(
        "percentage", F.round(F.col("count") / F.sum("count").over(W.partitionBy()), 3)
    )
    .withColumn("all", F.lit("all"))
).display()

# COMMAND ----------

# MAGIC %md ## Combine data

# COMMAND ----------

alloc_combined = (
    alloc
    # .join(food_segmentation.select("cust_id", "segment_desc"), on = "cust_id", how = "left")
    .join(
        segtco_history_.select("cust_id", "cust_band_fd"), on="cust_id", how="left"
    ).join(
        ss_result.select(
            "cust_id",
            F.col("estimated_spend").alias("estimated_spend_ss"),
            F.col("estimated_headroom").alias("estimated_headroom_ss"),
            F.col("spend_plus_headroom").alias("spend_plus_headroom_ss"),
        ),
        how="left",
        on="cust_id",
    )
)
alloc_combined = alloc_combined.withColumn(
    "headroom_ratio", F.col("estimated_headroom") / F.col("estimated_spend")
)

alloc_combined.count()
alloc_combined.display()

# COMMAND ----------

(
    alloc_combined.groupBy("desc")
    .count()
    .withColumn(
        "percentage", F.round(F.col("count") / F.sum("count").over(W.partitionBy()), 3)
    )
    .withColumn("all", F.lit("all"))
).display()

# COMMAND ----------

alloc_combined.filter(F.col("estimated_spend") <= F.col("estimated_spend_ss")).count()

# COMMAND ----------

alloc_combined.filter(
    F.col("estimated_headroom") <= F.col("estimated_headroom_ss")
).count()

# COMMAND ----------

alloc_combined.filter(
    F.col("spend_plus_headroom") < F.col("spend_plus_headroom_ss")
).count()

# COMMAND ----------

alloc_combined.filter(
    F.col("spend_plus_headroom") <= F.col("spend_plus_headroom_ss")
).count()

# COMMAND ----------

alloc_combined.filter(
    F.col("spend_plus_headroom") <= F.col("spend_plus_headroom_ss")
).groupBy("desc", "cust_band_fd").count().display()

# COMMAND ----------

# MAGIC %md ## offer by TCOL

# COMMAND ----------

# alloc_combined.groupBy('desc', 'cust_band_fd').count()\
#   .withColumn('percentage', F.round(F.col('count') / F.sum('count')\
#   .over(W.partitionBy()),3)).display()
alloc_combined.groupBy("desc", "cust_band_fd").count().display()

# COMMAND ----------

# MAGIC %md # Analysis

# COMMAND ----------

# MAGIC %md ## how many users have null segments?

# COMMAND ----------

# %sql select * from loyalty_azlab_prod.predictions_p_tbl where campaign = 20230519

# predictions = spark.sql("select * from loyalty_azlab_prod.predictions_p_tbl where campaign = 20230519 ")
# predictions = (
#   predictions
#   .select("cust_id", "50percentile_time_window_max_spend_basket",
#   "75percentile_time_window_max_spend_basket", "85percentile_time_window_max_spend_basket",
#   "90percentile_time_window_max_spend_basket")
#   .groupby("cust_id")
#   .sum()
# )
# predictions.display()

# %sql select count (distinct cust_id) from loyalty_azlab_prod.predictions_p_tbl where campaign = 20230519 and (experian_hh_composition is null or segmentation is null)

# prediction = spark.sql("select * from loyalty_azlab_prod.predictions_p_tbl where campaign = 20230519")

# prediction.filter((F.col("experian_hh_composition").isNull()) | (F.col("segmentation").isNull())).select("cust_id").distinct().count()

# COMMAND ----------

# MAGIC %md ## without the null segment, how many distinct segment are there per user
# MAGIC - everyone have unqiue segmentation

# COMMAND ----------

# prediction.filter((~F.col("experian_hh_composition").isNull()) & (~F.col("segmentation").isNull())).select("cust_id").distinct().count()

# display(
#   prediction.select("cust_id", "experian_hh_composition", "segmentation")
#   .distinct()
#   .filter((~F.col("experian_hh_composition").isNull()) & (~F.col("segmentation").isNull()))
#   .groupby("cust_id", "experian_hh_composition", "segmentation")
#   .count())

# display(
#   prediction.select("cust_id", "experian_hh_composition", "segmentation")
#   .distinct()
#   .filter((~F.col("experian_hh_composition").isNull()) & (~F.col("segmentation").isNull()))
#   .groupby("cust_id", "experian_hh_composition", "segmentation")
#   .count()
#   .groupby('count')
#   .count())

# COMMAND ----------

# MAGIC %md ## Estimated spend of 0
# MAGIC Around 3 million customers have an spend less than 5, this is very low, so it is not going to be that useful as the recommendation requires people to have spend money before. (£1, £2, and £5 have around the same amount of customers )

# COMMAND ----------

spend_less_than_5 = alloc_combined.filter(F.col("estimated_spend") < 5)
spend_less_than_5.count()

# COMMAND ----------

(alloc_combined.filter(F.col("estimated_spend") < 10)).count()


# COMMAND ----------

(alloc_combined.filter(F.col("estimated_spend") < 20)).count()


# COMMAND ----------

(alloc_combined.filter(F.col("estimated_spend") < 30)).count()


# COMMAND ----------

spend_less_than_5.groupby("desc").count().display()

# COMMAND ----------

fig = px.histogram(
    spend_less_than_5.select("estimated_headroom").toPandas(),
    x="estimated_headroom",
    nbins=20,
)
fig.show()

# COMMAND ----------

# MAGIC %md ## Segmentaiton
# MAGIC what is the overlap between TCOL and food segmentation

# COMMAND ----------

# MAGIC %md
# MAGIC - What is the minimun spend needed to get the offer
# MAGIC
# MAGIC - What is the markdown of the offer
# MAGIC
# MAGIC - Ratio of minimun spend / markdown for the offer,  if its >1 its good
# MAGIC If its < 1 means losing money

# COMMAND ----------

# alloc_spend_gt_x = (alloc_combined.filter(F.col("estimated_spend") >= 5 ))

# offer_dict = {'16388': "£3 off when you spend £20 on M&S food in store",
#                     '14140': "£5 off when you spend £30 on M&S food in store",
#                     '14141': "£5 off when you spend £40 on M&S food in store",
#                     '14142': "£5 off when you spend £50 on M&S food in store",
#                     '14238': "£7 off when you spend £70 on M&S food in store",
#                     '14144': "£9 off when you spend £90 on M&S food in store",
#                     '14239': "£10 off when you spend £100 on M&S food in store",
#                     '14146': "£12 off when you spend £120 on M&S food in store",
#                     '14242': "£14 off when you spend £140 on M&S food in store",
#                     '14190': "£16 off when you spend £160 on M&S food in store",
#                     '14191': "£20 off when you spend £200 on M&S food in store"
# }

# offer_value = pd.DataFrame({'offer_id': [i for i in offer_dict.keys()],
# "markdown": [offer_dict[i].split(" off")[0].replace("£", "") for i in offer_dict.keys()],
# "offer_spend_value":[offer_dict[i].split("spend ")[1].split(" on")[0].replace("£", "") for i in offer_dict.keys()]
# })
# offer_value = spark.createDataFrame(offer_value)
# offer_value.display()

# alloc_spend_gt_x = (alloc_spend_gt_x.join(offer_value, on = "offer_id", how = "left"))

# alloc_spend_gt_x = (
#   alloc_spend_gt_x
#   .withColumn("spend_needed_to_redeem", F.col("offer_spend_value") - F.col("spend_plus_headroom"))
#   .withColumn("spend_needed_ex_headroom", F.col("offer_spend_value") - F.col("estimated_spend"))
#   .withColumn("spend_to_markdown_ratio", F.col("spend_needed_to_redeem") / F.col("markdown"))
#   .withColumn("spend_to_markdown_ratio_exc_headroom", F.col("spend_needed_ex_headroom") / F.col("markdown"))
# )


# COMMAND ----------

# alloc_spend_gt_x.groupby("segment_desc", "cust_band_fd").count().display()

# COMMAND ----------

# alloc_spend_gt_x.groupBy('desc').count()\
#   .withColumn('percentage', F.round(F.col('count') / F.sum('count')\
#   .over(W.partitionBy()),3)).display()

# COMMAND ----------

# MAGIC %md ## Ratio of headroom / spend

# COMMAND ----------

# MAGIC %md how many have 20% factor

# COMMAND ----------

alloc_combined.count(), alloc_combined.filter(F.col("headroom_ratio") == 0.2).count()

# COMMAND ----------

alloc_combined.groupby("cust_band_fd").count().display()

# COMMAND ----------

alloc_combined.filter(F.col("headroom_ratio") == 0.2).groupby(
    "cust_band_fd"
).count().display()

# COMMAND ----------

# MAGIC %md ##  TCOL

# COMMAND ----------

display(alloc_combined)

# COMMAND ----------

# MAGIC %md ### percentile of spend by TCOL

# COMMAND ----------


def get_expr_agg(col, pct_list=(10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 100)):
    """
    method to fetch the statistics at defined percentile levels + the mean
    """
    out_expr = [F.mean(col).cast(T.DoubleType()).alias(f"average_{col}")]
    for p in pct_list:
        pct = float(p / 100.0)
        out_expr.append(
            F.expr(f"percentile_approx({col}, {pct})")
            .cast(T.DoubleType())
            .alias(f"{p}percentile_{col}")
        )
    return out_expr


# COMMAND ----------

alloc_combined.groupBy("cust_band_fd").agg(*get_expr_agg("estimated_spend")).display()

# COMMAND ----------

spend_df = spark.createDataFrame(
    pd.melt(
        alloc_combined.groupBy("cust_band_fd")
        .agg(*get_expr_agg("estimated_spend"))
        .toPandas(),
        id_vars="cust_band_fd",
    )
)
# spend_df.display()


# COMMAND ----------

spend_df.filter(F.col("variable") != "100percentile_estimated_spend").display()

# COMMAND ----------

# MAGIC %md ### percentile of headroom by tcol

# COMMAND ----------

alloc_combined.groupBy("cust_band_fd").agg(
    *get_expr_agg("estimated_headroom")
).display()

# COMMAND ----------

headroom_df = spark.createDataFrame(
    pd.melt(
        alloc_combined.groupBy("cust_band_fd")
        .agg(*get_expr_agg("estimated_headroom"))
        .toPandas(),
        id_vars="cust_band_fd",
    )
)

# COMMAND ----------

headroom_df.filter(F.col("variable") != "100percentile_estimated_headroom").display()

# COMMAND ----------

# MAGIC %md ### percentile of spend plus headroom by tcol

# COMMAND ----------

alloc_combined.groupBy("cust_band_fd").agg(
    *get_expr_agg("spend_plus_headroom")
).display()

# COMMAND ----------

spend_plus_headroom = spark.createDataFrame(
    pd.melt(
        alloc_combined.groupBy("cust_band_fd")
        .agg(*get_expr_agg("spend_plus_headroom"))
        .toPandas(),
        id_vars="cust_band_fd",
    )
)
spend_plus_headroom.display()


# COMMAND ----------

spend_plus_headroom.filter(
    F.col("variable") != "100percentile_spend_plus_headroom"
).display()

# COMMAND ----------

# MAGIC %md ### headroom ratio

# COMMAND ----------

headroom_ratio_df = spark.createDataFrame(
    pd.melt(
        alloc_combined.groupBy("cust_band_fd")
        .agg(*get_expr_agg("headroom_ratio"))
        .toPandas(),
        id_vars="cust_band_fd",
    )
)
headroom_ratio_df.filter(F.col("variable") != "100percentile_headroom_ratio").display()

# COMMAND ----------

# MAGIC %md # VIP list

# COMMAND ----------


alloc = spark.read.parquet(
    "/mnt/centralds/offerallocation/headroom/analysis/230720/allocation_accumulator"
)
alloc.count()

# COMMAND ----------

alloc.groupBy("desc").count().withColumn(
    "percentage", F.round(F.col("count") / F.sum("count").over(W.partitionBy()), 3)
).display()

# COMMAND ----------

vip = spark.read.csv(
    "dbfs:/mnt/centralds/offerallocation/TMO/vip_customers/vip_list_20221201",
    header=True,
)

sparks = spark.sql(
    "select account_id, cust_id from analytics_trans_prod.sparks_account"
)
display(vip.join(sparks, how="left", on="account_id"))

# COMMAND ----------

display(
    alloc.join(vip.join(sparks, how="left", on="account_id"), how="inner", on="cust_id")
)

# COMMAND ----------

# %sql select * from analytics_trans_prod.sparks_account

# COMMAND ----------

# Charlotte
display(alloc.filter(F.col("cust_id") == 4585580942257894003))

# COMMAND ----------

# Dapeng
display(alloc.filter(F.col("cust_id") == 6872732896086312431))

# COMMAND ----------

# Sherry
display(alloc.filter(F.col("cust_id") == "-7385606211536121860"))

# COMMAND ----------

# Andy
display(alloc.filter(F.col("cust_id") == "-7117536182747671208"))

# COMMAND ----------

# %sql select * from loyalty_azlab_prod.predictions_230522_p_tbl where cust_id = -7117536182747671208

# COMMAND ----------

# MAGIC %md # Investigate individual

# COMMAND ----------

# MAGIC %sql select * from loyalty_azlab_prod.predictions_p_tbl where campaign = 20230720 and cust_id = -7117536182747671208

# COMMAND ----------


# COMMAND ----------

# MAGIC %md # Investigate feature

# COMMAND ----------

# MAGIC %md does the basket spend always below the weekly spend?
# MAGIC Individually no, but when summing up at l2 id level, then the basket is always below the timewindow spend

# COMMAND ----------

# MAGIC %sql select * from loyalty_azlab_prod.headroom_etl_data_np_p_tbl where campaign = 20230724

# COMMAND ----------

# MAGIC %sql select * from loyalty_azlab_prod.headroom_etl_data_np_p_tbl where campaign = 20230724 and cust_id = 6872732896086312431

# COMMAND ----------

# MAGIC %sql select count(*) from loyalty_azlab_prod.headroom_etl_data_np_p_tbl  where campaign = 20230724

# COMMAND ----------

# MAGIC %sql select count(*) from loyalty_azlab_prod.headroom_etl_data_np_p_tbl  where campaign = 20230724 and l2_id_total_spend_basket > l2_id_total_time_window_spend

# COMMAND ----------

# MAGIC %sql select count(*) from loyalty_azlab_prod.headroom_etl_data_np_p_tbl where campaign = 20230720 and  l2_id_total_spend_basket >= l2_id_total_time_window_spend

# COMMAND ----------

# MAGIC %sql select count(*) from loyalty_azlab_prod.headroom_etl_data_np_p_tbl where campaign = 20230720 and  l2_id_total_spend_basket <l2_id_total_time_window_spend

# COMMAND ----------

# MAGIC %sql select count(*) from loyalty_azlab_prod.headroom_etl_data_np_p_tbl where campaign = 20230720 and  l2_id_total_spend_basket = l2_id_total_time_window_spend

# COMMAND ----------

# MAGIC %sql select * from loyalty_azlab_prod.headroom_etl_data_np_p_tbl where campaign = 20230720 and  l2_id_total_spend_basket >= l2_id_total_time_window_spend

# COMMAND ----------

etl = spark.sql(
    "select * from loyalty_azlab_prod.headroom_etl_data_np_p_tbl where campaign = 20230720"
)
etl_2 = etl.groupby("cust_id").agg(
    F.sum("l2_id_total_spend_basket").alias("total_basket_spend"),
    F.sum("l2_id_total_time_window_spend").alias("total_time_window_spend"),
)
etl_2.display()

# COMMAND ----------

etl_2.filter(F.col("total_basket_spend") > F.col("total_basket_spend")).count()

# COMMAND ----------

etl.display()

# COMMAND ----------

etl.groupby("cust_id").agg(F.sum("visits").alias("upper_bound_visits")).filter(
    F.col("upper_bound_visits") < 2
).count()

# COMMAND ----------

# MAGIC %md #Visits

# COMMAND ----------

# MAGIC %md for customer who have low estimated spend, how many times did they shop?

# COMMAND ----------

etl_visits = spark.sql(
    "select distinct cust_id, count_user_basket, count_user_time_window from loyalty_azlab_prod.headroom_etl_data_np_p_tbl where campaign = 20230724"
)
etl_visits.cache()
etl_visits.count()

# COMMAND ----------

visits = alloc_combined.join(etl_visits, how="left", on="cust_id")
visits.display()

# COMMAND ----------

# for spend less than 30  what was the number of visits by tcol
(
    visits.filter(F.col("spend_plus_headroom") < 30)
    .filter(F.col("count_user_basket") < 5)
    .groupby("cust_band_fd", "count_user_basket")
    .count()
).display()

# COMMAND ----------

# for spend less than 30  what was the number of visits by tcol
(
    visits.filter(F.col("spend_plus_headroom") < 30)
    .filter(F.col("count_user_time_window") < 10)
    .groupby("cust_band_fd", "count_user_time_window")
    .count()
).display()

# COMMAND ----------

(
    visits.filter(F.col("spend_plus_headroom") < 20)
    .filter(F.col("count_user_basket") < 3)
    .groupby("desc", "cust_band_fd", "count_user_basket")
    .count()
).display()

# COMMAND ----------

visits.filter(F.col("spend_plus_headroom") < 20).groupby(
    "desc", "cust_band_fd", "count_user_time_window"
).count().display()

# COMMAND ----------

# MAGIC %md for customer who had few baskets, what was the offers given?

# COMMAND ----------

visits.filter(F.col("count_user_basket") <= 2).groupby(
    "cust_band_fd", "desc"
).count().display()

# COMMAND ----------

# MAGIC %md for customer who had few time window ind, what was the offers given?

# COMMAND ----------

visits.filter(F.col("count_user_time_window") < 2).groupby(
    "cust_band_fd", "desc"
).count().display()

# COMMAND ----------

visits.filter(F.col("count_user_time_window").isNull()).count()

# COMMAND ----------

visits.filter(F.col("count_user_basket").isNull()).count()

# COMMAND ----------


# COMMAND ----------


# COMMAND ----------
