# Databricks notebook source
import os
import random
from datetime import datetime, timedelta
from functools import partial, reduce

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import cm
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from pyspark.ml.feature import PCA as sparkPCA
from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql import types as T
from pyspark.sql.window import Window
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score

sns.set_style("darkgrid")

# COMMAND ----------

# DBTITLE 1,Dedupe Recapture customers
recapture = spark.sql(
    "select * from ciu_azlab_dev.20211221_ns_recapture_campaign_agg_final_offer where fd_offer_eligible='Y'"
)

# display(recapture)
recapture.select("sparks_account_id").distinct().count()

# COMMAND ----------

allocation_path = (
    "dbfs:/mnt/centralds/offerallocation/headroom/prod_run/export/202112/export_v3"
)
allocation_temp_path = f"{allocation_path}_temp"

# COMMAND ----------

sparks_account_df = spark.table("analytics_trans_prod.sparks_account")

customers_path = (
    "dbfs:/mnt/centralds/offerallocation/headroom/prod_run/prod_input/202112/customers"
)
pred_path = "dbfs:/mnt/centralds/offerallocation/headroom/development/v0.2/prediction/202112/v0.0.0/predictions"

customers = spark.read.parquet(customers_path)

pred = (
    spark.read.parquet(pred_path)
    .join(sparks_account_df.select("cust_id", "account_id").distinct(), on="cust_id")
    .join(customers.select("account_id").distinct(), on="account_id")
    # Deduping against the recaptuure campaign
    .join(
        recapture.select(F.col("sparks_account_id").alias("account_id")).distinct(),
        on="account_id",
        how="leftanti",
    )
)


def days_between(d1, d2, date_format="%Y%m%d"):
    d1 = datetime.strptime(str(d1), date_format)
    d2 = datetime.strptime(str(d2), date_format)
    return abs((d2 - d1).days)


prediction_time_range = [20210101, 20211201]
prediction_time_span = days_between(prediction_time_range[1], prediction_time_range[0])
prediction_week = prediction_time_span / 7.0
prediction_month = prediction_time_span / 30.0

hhs = (
    pred.select("experian_hh_composition").distinct().rdd.map(lambda x: x[0]).collect()
)

display(pred)

# COMMAND ----------

pred.select("account_id").distinct().count()

# COMMAND ----------

pred_scores = (
    pred.withColumn(
        "pct_error",
        100.0
        * (F.col("prediction_out") - F.col("total_spend"))
        / (F.col("total_spend")),
    )
    .groupby("account_id", "experian_hh_composition", "segmentation")
    .agg(
        F.sum("total_spend").alias("sum_total_spend"),
        (F.sum("total_spend") / prediction_month).alias("sum_total_spend_month"),
        (F.sum("total_spend") / prediction_week).alias("sum_total_spend_week"),
        F.sum("prediction_out").alias("sum_prediction"),
        (F.sum("prediction_out") / prediction_month).alias("sum_prediction_month"),
        (F.sum("prediction_out") / prediction_week).alias("sum_prediction_week"),
        F.mean("pct_error").alias("mean_pct_error"),
    )
    .withColumn("offer_id", F.lit(None))
)

display(pred_scores)

# COMMAND ----------

display(
    pred_scores.filter(
        (F.col("mean_pct_error") >= 0) & (F.col("mean_pct_error") <= 100)
    )
)

# COMMAND ----------

display(
    pred_scores.filter(
        (F.col("mean_pct_error") >= 0) & (F.col("mean_pct_error") <= 100)
    )
)

# COMMAND ----------

pred_scores.filter(
    (F.col("mean_pct_error") >= 0) & (F.col("mean_pct_error") <= 100)
).count()

# COMMAND ----------

group1 = (
    pred_scores
    #           .filter((F.col("mean_pct_error")>=5) & (F.col("mean_pct_error")<=20))
    .withColumn(
        "outlier",
        F.when(
            (F.col("mean_pct_error") >= 5) & (F.col("mean_pct_error") <= 20), 0
        ).otherwise(1),
    )
)


# COMMAND ----------

group1.select("account_id").distinct().count()

# COMMAND ----------

# 2. Save £5 when you spend £40
# 3. Save £5 when you spend £50
# 4. Save £7 when you spend £70
# 5. Save £9 when you spend £90
# 6. Save £10 when you spend £100
# 7. Save £12 when you spend £120
# 8. Save £14 when you spend £140

print(
    """Offers:
14055    £5 off when you spend £30 on M&S food in store
13992    £5 off when you spend £40 on M&S food in store
13993    £5 off when you spend £50 on M&S food in store
13994    £7 off when you spend £70 on M&S food in store
13995    £9 off when you spend £90 on M&S food in store
13996    £10 off when you spend £100 on M&S food in store
13997    £12 off when you spend £120 on M&S food in store
14241    £14 off when you spend £140 on M&S food in store
14183    £16 off when you spend £160 on M&S food in store
14184    £20 off when you spend £200 on M&S food in store
"""
)

offer_desc = {
    14055: "£5 off when you spend £30 on M&S food in store",
    13992: "£5 off when you spend £40 on M&S food in store",
    13993: "£5 off when you spend £50 on M&S food in store",
    13994: "£7 off when you spend £70 on M&S food in store",
    13995: "£9 off when you spend £90 on M&S food in store",
    13996: "£10 off when you spend £100 on M&S food in store",
    13997: "£12 off when you spend £120 on M&S food in store",
    14241: "£14 off when you spend £140 on M&S food in store",
    14183: "£16 off when you spend £160 on M&S food in store",
    14184: "£20 off when you spend £200 on M&S food in store",
}

# COMMAND ----------

offer_lims = {
    14055: [0, 30],
    13992: [30, 40],
    13993: [40, 50],
    13994: [50, 70],
    13995: [70, 90],
    13996: [90, 100],
    13997: [100, 120],
    14241: [120, 140],
    14183: [140, 160],
    14184: [180, 220],
}

large_lim = 220
large_offers = [14241, 14183, 14184]
small_offers = [14055, 13992, 13993]


def get_offer(rand, offers):
    idx = int(rand * len(offers))
    return offers[idx]


get_large_offer = F.udf(partial(get_offer, offers=large_offers), T.IntegerType())
get_small_offer = F.udf(partial(get_offer, offers=small_offers), T.IntegerType())


@F.udf(T.StringType())
def get_offer_desc(offer):
    return offer_desc[offer]


@F.udf(T.IntegerType())
def get_offer_upper_lim(offer):
    return offer_lims[offer][1]


group1_allocation = group1

headroom_factor = 1.15  # 15% increase
max_increase = 80
min_increase = -50


group1_allocation = (
    group1_allocation.withColumn(
        "used_headroom_frac",
        F.when(F.col("mean_pct_error") >= max_increase, (1.0 + max_increase / 100.0))
        .when(F.col("mean_pct_error") <= min_increase, (1.0 + min_increase / 100.0))
        .otherwise(1.0 + F.col("mean_pct_error") / 100.0),
    )
    .withColumn(
        "total_used_headroom",
        F.col("sum_total_spend_month") * F.col("used_headroom_frac"),
    )
    .withColumn("rand", F.rand())
)


for k, v in offer_lims.items():
    offer_id = int(k)
    group1_allocation = group1_allocation.withColumn(
        "offer_id",
        F.when(
            (F.col("total_used_headroom") >= v[0])
            & (F.col("total_used_headroom") < v[1]),
            offer_id,
        ).otherwise(F.col("offer_id")),
    )

group1_allocation.write.parquet(allocation_temp_path, mode="overwrite")
group1_allocation_temp = spark.read.parquet(allocation_temp_path)


group1_allocation_final = (
    group1_allocation_temp
    # If very large headroom. Probably some outliers. For now random spread these offers over the top offer range.
    .withColumn(
        "offer_id",
        F.when(
            (F.col("total_used_headroom") >= large_lim), get_large_offer(F.col("rand"))
        ).otherwise(F.col("offer_id")),
    )
    # If offer Id is still null then an outlier. Give a random small offer.
    .withColumn(
        "offer_id",
        F.when((F.col("offer_id").isNull()), get_small_offer(F.col("rand"))).otherwise(
            F.col("offer_id")
        ),
    )
    .withColumn("desc", get_offer_desc(F.col("offer_id")))
    .withColumn("upper_lim", get_offer_upper_lim(F.col("offer_id")))
)


display(group1_allocation_final)

# COMMAND ----------

display(group1_allocation_final)

# COMMAND ----------

display(
    group1_allocation_final.filter(F.col("offer_id").isNull())
    #         .filter(F.col("offer_id")==8)
)

# COMMAND ----------

display(
    group1_allocation_final.groupby("offer_id", "desc", "upper_lim")
    .count()
    .orderBy("upper_lim")
)

# COMMAND ----------

# add flag
custs = spark.read.parquet(
    "dbfs:/mnt/centralds/offerallocation/headroom/prod_run/prod_input/202112/customers_v3"
)
custs_recent = (
    custs.filter(F.col("recent_purchase") == 1).select("account_id").distinct()
)


group1_allocation_final_out = (
    group1_allocation_final.join(
        custs_recent.withColumn("analysis_flag", F.lit(1)), on="account_id", how="left"
    )
    .fillna(0, subset=["analysis_flag"])
    .select(out_columns)
)

out_columns = ["account_id", "offer_id", F.lit(2).alias("test_group"), "analysis_flag"]
group1_allocation_final_out.coalesce(1).write.csv(
    allocation_path, header=True, mode="overwrite"
)

# COMMAND ----------

group1_allocation_final_out.select("account_id").distinct().count()

# COMMAND ----------

# DBTITLE 1,3 Month Analysis Flag
display(group1_allocation_final_out.groupby("test_group", "analysis_flag").count())

# COMMAND ----------

group1_allocation_temp = spark.read.parquet(allocation_temp_path)
display(group1_allocation_temp.filter(F.col("account_id") == 800100000450))

# COMMAND ----------
