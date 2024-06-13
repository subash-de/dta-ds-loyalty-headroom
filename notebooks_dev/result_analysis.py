# Databricks notebook source
# MAGIC %run ./bootstrap

# COMMAND ----------

config

# COMMAND ----------

import inspect

lines = inspect.getsource(TransactionsManager.add_time_window_ind)
print(lines)

# COMMAND ----------

# import inspect
# lines = inspect.getsource(TransactionsManager.get)
# print(lines)

# COMMAND ----------

# MAGIC %md # Result analysis

# COMMAND ----------

# MAGIC %md
# MAGIC - compare the prod allocation with time based apporach

# COMMAND ----------

import seaborn as sns
from offerallocationv2.utils import tmo_utils
from pyspark.sql import Column, DataFrame
from pyspark.sql import Window as W
from pyspark.sql import functions as F
from pyspark.sql import types as T

# COMMAND ----------

# MAGIC %md # Prod 2023 04 24

# COMMAND ----------

prod_230424 = spark.sql(
    "select * from datascienceoffers_analyse_prod.headroom_allocation_p_tbl where campaign = 20230424"
)
display(prod_230424)

# COMMAND ----------

prod_230424.groupBy("desc").count().withColumn(
    "percentage", F.round(F.col("count") / F.sum("count").over(W.partitionBy()), 3)
).display()

# COMMAND ----------

# MAGIC %md # Lab 2023 04 24

# COMMAND ----------

lab_230424 = spark.sql(
    "select * from loyalty_azlab_prod.headroom_allocation_p_tbl where campaign = 20230424"
)
display(lab_230424)

# COMMAND ----------

lab_230424.groupBy("desc").count().withColumn(
    "percentage", F.round(F.col("count") / F.sum("count").over(W.partitionBy()), 3)
).display()

# COMMAND ----------

# MAGIC %md # Segmentation

# COMMAND ----------

segtco_history_df = spark.table("customer_azbase_prod.segtco_history")
segtco_history_ = tmo_utils.get_preceding_segtco_history(segtco_history_df, 20230424)

# COMMAND ----------

# display(segtco_history_)

# COMMAND ----------

# MAGIC %md # What is the movement for the different segments

# COMMAND ----------

lab_segment = lab_230424.join(
    segtco_history_.select("cust_id", "cust_band_fd"), on="cust_id", how="left"
)
display(lab_segment)

# COMMAND ----------

lab_segment.groupby("cust_band_fd", "desc").count().display()

# COMMAND ----------

lab_segment.crosstab("prod_desc", "cust_band_fd").display()

# COMMAND ----------

# MAGIC %md # What is the movement between different offer levels?

# COMMAND ----------

alloc = prod_230424.select("cust_id", F.col("desc").alias("prod_desc")).join(
    lab_230424.select("cust_id", F.col("desc").alias("lab_desc")),
    on="cust_id",
    how="outer",
)
display(alloc)

# COMMAND ----------

alloc.groupby("prod_desc", "lab_desc").count().display()

# COMMAND ----------

alloc.crosstab("prod_desc", "lab_desc").display()

# COMMAND ----------


# COMMAND ----------

# MAGIC %md # distribution of spend vs distribution of headroom

# COMMAND ----------

display(lab_segment)

# COMMAND ----------


# COMMAND ----------

# MAGIC %md # For each desc,  whats is the destribution of spend, and headroom

# COMMAND ----------

display(lab_segment)

# COMMAND ----------

capped_df = lab_segment.withColumn(
    "estimated_spend_capped",
    F.when(F.col("estimated_spend") > 200, 200).otherwise(F.col("estimated_spend")),
).withColumn(
    "estimated_headroom_capped",
    F.when(F.col("estimated_headroom") > 200, 200).otherwise(
        F.col("estimated_headroom")
    ),
)
capped_df.display()


# COMMAND ----------

capped_pd = capped_df.select(
    "desc", "cust_band_fd", "estimated_spend_capped", "estimated_headroom_capped"
).toPandas()

# COMMAND ----------

capped_pd

# COMMAND ----------

sns.kdeplot(
    data=capped_pd,
    x="estimated_spend_capped",
    y="estimated_headroom_capped",
    hue="desc",
)

# COMMAND ----------

sns.kdeplot(data=capped_pd, x="estimated_spend_capped", hue="desc")


# COMMAND ----------


# COMMAND ----------


# COMMAND ----------

# MAGIC %sql select case when customer_segment = 1 then 'grab and goers' else 'Err' end as segment_desc, account_id from fci_azlab_dev.sg_segmentation_cust_base_scores_all where year_end = 20230401

# COMMAND ----------

# MAGIC %sql select * from fci_azlab_dev.sg_segmentation_cust_base_scores_all where year_end = 20230401

# COMMAND ----------

# MAGIC %sql select case
# MAGIC
# MAGIC         when customer_segment = 1 then 'grab and goers'
# MAGIC
# MAGIC         when customer_segment = 2 then 'magic seekers'
# MAGIC
# MAGIC         when customer_segment = 3 then 'basket builders'
# MAGIC
# MAGIC         when customer_segment = 4 then 'easy eaters'
# MAGIC
# MAGIC         when customer_segment = 5 then 'savvy savers'
# MAGIC
# MAGIC         when customer_segment = 6 then 'social shoppers'
# MAGIC
# MAGIC         else 'Err' end as segment_desc, account_id
# MAGIC
# MAGIC from
# MAGIC
# MAGIC fci_azlab_dev.sg_segmentation_cust_base_scores_all
# MAGIC
# MAGIC where year_end = 20230401

# COMMAND ----------

# MAGIC %sql select * from fci_azlab_dev.ls_food_segment_20230429

# COMMAND ----------

# MAGIC %sql select * from fci_azlab_dev.ls_food_segment_20230429
