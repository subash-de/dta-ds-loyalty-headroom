# Databricks notebook source
# MAGIC %md "feature_col": "85percentile_time_window_max_spend_basket"

# COMMAND ----------

# MAGIC %run ./bootstrap

# COMMAND ----------

config

# COMMAND ----------

# from offerallocationv2.utils import tmo_utils
import pandas as pd 
import plotly
import plotly.express as px
from pyspark.sql import functions as F, DataFrame, Column, Window as W, types as T
import seaborn as sns


# COMMAND ----------

# MAGIC %md # Data 

# COMMAND ----------

# MAGIC %md ## Food segmentation 
# MAGIC - using fci_azlab_dev.ls_food_segment_20230429 instead of fci_azlab_dev.sg_segmentation_cust_base_scores_all for segmentation, was told there was some issue with the original table 

# COMMAND ----------

# %sql select * from fci_azlab_dev.ls_food_segment_20230429

# COMMAND ----------

# food_segmentation = spark.sql("select * from fci_azlab_dev.ls_food_segment_20230429")
# food_segmentation.count()
# food_segmentation.write.parquet("/mnt/centralds/offerallocation/headroom/analysis/230426/food_segmentation_230429")

food_segmentation = spark.read.parquet("/mnt/centralds/offerallocation/headroom/analysis/230426/food_segmentation_230429")
food_segmentation.count()

# COMMAND ----------

food_segmentation = (
  food_segmentation
  .withColumn("segment_desc", F.when(F.col("customer_segment") == 1, "1_grab_and_goers")
  .when(F.col("customer_segment") == 2, "2_magic_seekers")
  .when(F.col("customer_segment") == 3, "3_basket_builders")
  .when(F.col("customer_segment") == 4, "4_easy_eaters")
  .when(F.col("customer_segment") == 5, "5_savvy_savers")
  .when(F.col("customer_segment") == 6, "6_social_shoppers")
  .otherwise("err"))
)
food_segmentation.display()

# COMMAND ----------

food_segmentation.groupby("segment_desc").count().orderBy(F.col("segment_desc")).display()

# COMMAND ----------

# MAGIC %md ## TCOL segmentation 

# COMMAND ----------

# MAGIC %md tcol segmentation, saving the file, as it takes a long time to load this data each time

# COMMAND ----------

# segtco_history_df = spark.table("customer_azbase_prod.segtco_history")
# segtco_history_ = tmo_utils.get_preceding_segtco_history(segtco_history_df, 20230426)
# segtco_history_.write.parquet("/mnt/centralds/offerallocation/headroom/analysis/230426/tcol_segmentation_230426")

# COMMAND ----------

segtco_history_ = spark.read.parquet("/mnt/centralds/offerallocation/headroom/analysis/230426/tcol_segmentation_230426")
segtco_history_.count()

# COMMAND ----------

segtco_history_.display()

# COMMAND ----------

# MAGIC %md ## Lab 2023 04 26

# COMMAND ----------

# %sql select * from loyalty_azlab_prod.headroom_allocation_p_tbl 
# %sql select campaign, count (distinct campaign) from loyalty_azlab_prod.headroom_allocation_p_tbl group by campaign  

# COMMAND ----------

# MAGIC %md  Saving the allocation result, because the allocation gets overwritten each time

# COMMAND ----------

# alloc = spark.sql("select * from loyalty_azlab_prod.headroom_allocation_p_tbl")
# # # alloc.groupby("campaign").count().show()
# alloc.write.parquet("/mnt/centralds/offerallocation/headroom/analysis/230515/allocation", mode = "overwrite")


# COMMAND ----------

# alloc = spark.sql("select * from loyalty_azlab_prod.headroom_allocation_p_tbl")
# alloc.write.parquet("/mnt/centralds/offerallocation/headroom/analysis/230426/allocation")

alloc = spark.read.parquet("/mnt/centralds/offerallocation/headroom/analysis/230515/allocation")
alloc.count()

# COMMAND ----------

alloc.display()

# COMMAND ----------

# MAGIC %md ## Combine data 

# COMMAND ----------

alloc_combined = (
  alloc
  .join(food_segmentation.select("cust_id", "segment_desc"), on = "cust_id", how = "left")
  .join(segtco_history_.select("cust_id", "cust_band_fd"), on = "cust_id", how = "left")
)
alloc_combined.count()
alloc_combined.display()

# COMMAND ----------

(alloc_combined.groupBy('desc').count()\
  .withColumn('percentage', F.round(F.col('count') / F.sum('count')\
  .over(W.partitionBy()),3)
)
.withColumn("all", F.lit("all"))
).display()

# COMMAND ----------

# MAGIC %md ## offer by segment 
# MAGIC
# MAGIC basket builder's offer are different compared with the other segments   
# MAGIC
# MAGIC Basket builder have more proporition of higher offers 

# COMMAND ----------

# alloc_combined.groupBy('desc', 'segment_desc').count()\
#   .withColumn('percentage', F.round(F.col('count') / F.sum('count')\
#   .over(W.partitionBy()),3)).display()
alloc_combined.groupBy('desc', 'segment_desc').count().display()

# COMMAND ----------

# MAGIC %md ## offer by TCOL

# COMMAND ----------

# alloc_combined.groupBy('desc', 'cust_band_fd').count()\
#   .withColumn('percentage', F.round(F.col('count') / F.sum('count')\
#   .over(W.partitionBy()),3)).display()
alloc_combined.groupBy('desc', 'cust_band_fd').count().display()

# COMMAND ----------

# MAGIC %md # Analysis 

# COMMAND ----------

# MAGIC %md ## how many users have null segments? 

# COMMAND ----------

# MAGIC %sql select * from loyalty_azlab_prod.predictions_p_tbl where campaign = 20230515 

# COMMAND ----------

predictions = spark.sql("select * from loyalty_azlab_prod.predictions_p_tbl where campaign = 20230515 ")
predictions = (
  predictions
  .select("cust_id", "50percentile_time_window_max_spend_basket", 
  "75percentile_time_window_max_spend_basket", "85percentile_time_window_max_spend_basket", 
  "90percentile_time_window_max_spend_basket")
  .groupby("cust_id")
  .sum()
)
predictions.display()

# COMMAND ----------

# MAGIC %sql select count (distinct cust_id) from loyalty_azlab_prod.predictions_p_tbl where campaign = 20230515 and (experian_hh_composition is null or segmentation is null) 

# COMMAND ----------

prediction = spark.sql("select * from loyalty_azlab_prod.predictions_p_tbl where campaign = 20230515")

# COMMAND ----------

prediction.filter((F.col("experian_hh_composition").isNull()) | (F.col("segmentation").isNull())).select("cust_id").distinct().count()

# COMMAND ----------

# MAGIC %md ## without the null segment, how many distinct segment are there per user 
# MAGIC - everyone have unqiue segmentation 

# COMMAND ----------

prediction.filter((~F.col("experian_hh_composition").isNull()) & (~F.col("segmentation").isNull())).select("cust_id").distinct().count()

# COMMAND ----------

display(
  prediction.select("cust_id", "experian_hh_composition", "segmentation")
  .distinct()
  .filter((~F.col("experian_hh_composition").isNull()) & (~F.col("segmentation").isNull()))
  .groupby("cust_id", "experian_hh_composition", "segmentation")
  .count())

# COMMAND ----------

display(
  prediction.select("cust_id", "experian_hh_composition", "segmentation")
  .distinct()
  .filter((~F.col("experian_hh_composition").isNull()) & (~F.col("segmentation").isNull()))
  .groupby("cust_id", "experian_hh_composition", "segmentation")
  .count()
  .groupby('count')
  .count())

# COMMAND ----------

# MAGIC %md ## Estimated spend of 0
# MAGIC Around 3 million customers have an spend less than 5, this is very low, so it is not going to be that useful as the recommendation requires people to have spend money before. (£1, £2, and £5 have around the same amount of customers )

# COMMAND ----------

spend_less_than_5 = (alloc_combined.filter(F.col("estimated_spend") < 5))
spend_less_than_5.count()

# COMMAND ----------

spend_less_than_5.groupby("desc").count().display()

# COMMAND ----------

fig = px.histogram(spend_less_than_5.select("estimated_headroom").toPandas(), x="estimated_headroom", nbins = 20)
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

alloc_spend_gt_x = (alloc_combined.filter(F.col("estimated_spend") >= 5 ))

offer_dict = {'16388': "£3 off when you spend £20 on M&S food in store",
                    '14140': "£5 off when you spend £30 on M&S food in store",
                    '14141': "£5 off when you spend £40 on M&S food in store",
                    '14142': "£5 off when you spend £50 on M&S food in store",
                    '14238': "£7 off when you spend £70 on M&S food in store",
                    '14144': "£9 off when you spend £90 on M&S food in store",
                    '14239': "£10 off when you spend £100 on M&S food in store",
                    '14146': "£12 off when you spend £120 on M&S food in store",
                    '14242': "£14 off when you spend £140 on M&S food in store",
                    '14190': "£16 off when you spend £160 on M&S food in store",
                    '14191': "£20 off when you spend £200 on M&S food in store"
}

offer_value = pd.DataFrame({'offer_id': [i for i in offer_dict.keys()],
"markdown": [offer_dict[i].split(" off")[0].replace("£", "") for i in offer_dict.keys()],
"offer_spend_value":[offer_dict[i].split("spend ")[1].split(" on")[0].replace("£", "") for i in offer_dict.keys()]
})
offer_value = spark.createDataFrame(offer_value) 
offer_value.display()

alloc_spend_gt_x = (alloc_spend_gt_x.join(offer_value, on = "offer_id", how = "left"))

alloc_spend_gt_x = (
  alloc_spend_gt_x
  .withColumn("spend_needed_to_redeem", F.col("offer_spend_value") - F.col("spend_plus_headroom"))
  .withColumn("spend_needed_ex_headroom", F.col("offer_spend_value") - F.col("estimated_spend"))
  .withColumn("spend_to_markdown_ratio", F.col("spend_needed_to_redeem") / F.col("markdown"))
  .withColumn("spend_to_markdown_ratio_exc_headroom", F.col("spend_needed_ex_headroom") / F.col("markdown"))
)


# COMMAND ----------

alloc_spend_gt_x.groupby("segment_desc", "cust_band_fd").count().display()

# COMMAND ----------

alloc_spend_gt_x.groupBy('desc').count()\
  .withColumn('percentage', F.round(F.col('count') / F.sum('count')\
  .over(W.partitionBy()),3)).display()

# COMMAND ----------



# COMMAND ----------

# MAGIC %md ## By Food Segmentation 

# COMMAND ----------

display(alloc_spend_gt_x.orderBy(F.rand()))

# COMMAND ----------

alloc_spend_gt_x.count()

# COMMAND ----------

# MAGIC %md how many negative spend, we shouldn't expect this unless its the top 2 offers 

# COMMAND ----------

# how many negative spend, we shouldn't expect this unless its the top 2 offers 
alloc_spend_gt_x.filter(F.col("spend_needed_to_redeem") <0).count()

# COMMAND ----------

alloc_spend_gt_x.filter(F.col("spend_needed_to_redeem") <0).groupby("desc").count().display()

# COMMAND ----------

# alloc_spend_gt_x.filter(F.col("spend_needed_to_redeem") <=0).filter(F.col("desc") == "£14 off when you spend £140 on M&S food in store").display()

# COMMAND ----------

alloc_spend_gt_x.groupby("segment_desc").count().display()

# COMMAND ----------

(alloc_spend_gt_x.groupby("segment_desc")
.agg(F.sum("spend_needed_to_redeem").alias("spend_needed_to_redeem"),
F.sum("spend_needed_ex_headroom").alias("spend_needed_ex_headroom"),
F.sum("markdown").alias("markdown"),)
.withColumn("SMR", F.col("spend_needed_to_redeem") / F.col("markdown"))
.withColumn("SMR_ex_headroom", F.col("spend_needed_ex_headroom") / F.col("markdown"))
).display()

# COMMAND ----------

(alloc_spend_gt_x.groupby("desc")
.agg(F.sum("spend_needed_to_redeem").alias("spend_needed_to_redeem"),
F.sum("spend_needed_ex_headroom").alias("spend_needed_ex_headroom"),
F.sum("markdown").alias("markdown"),)
.withColumn("SMR", F.col("spend_needed_to_redeem") / F.col("markdown"))
.withColumn("SMR_ex_headroom", F.col("spend_needed_ex_headroom") / F.col("markdown"))
).display()

# COMMAND ----------

(alloc_spend_gt_x.groupby("cust_band_fd")
.agg(F.sum("spend_needed_to_redeem").alias("spend_needed_to_redeem"),
F.sum("spend_needed_ex_headroom").alias("spend_needed_ex_headroom"),
F.sum("markdown").alias("markdown"),)
.withColumn("SMR", F.col("spend_needed_to_redeem") / F.col("markdown"))
.withColumn("SMR_ex_headroom", F.col("spend_needed_ex_headroom") / F.col("markdown"))
).display()

# COMMAND ----------



# COMMAND ----------

(alloc_spend_gt_x.groupby("segment_desc", "desc")
.agg(F.sum("spend_needed_to_redeem").alias("spend_needed_to_redeem"),
F.sum("spend_needed_ex_headroom").alias("spend_needed_ex_headroom"),
F.sum("markdown").alias("markdown"),
F.count("*").alias("count"))
.withColumn("SMR", F.col("spend_needed_to_redeem") / F.col("markdown"))
.withColumn("SMR_ex_headroom", F.col("spend_needed_ex_headroom") / F.col("markdown"))
.withColumn("avg_spend_to_red_ex_headroom", F.col("spend_needed_ex_headroom") / F.col("count"))
).display()

# COMMAND ----------

(alloc_spend_gt_x.groupby("segment_desc", "cust_band_fd")
.agg(F.sum("spend_needed_to_redeem").alias("spend_needed_to_redeem"),
F.sum("spend_needed_ex_headroom").alias("spend_needed_ex_headroom"),
F.sum("markdown").alias("markdown"),
F.count("*").alias("count"))
.withColumn("SMR", F.col("spend_needed_to_redeem") / F.col("markdown"))
.withColumn("SMR_ex_headroom", F.col("spend_needed_ex_headroom") / F.col("markdown"))
.withColumn("avg_spend_to_red_ex_headroom", F.col("spend_needed_ex_headroom") / F.col("count"))
).display()

# COMMAND ----------

(alloc_spend_gt_x.groupby("cust_band_fd", "desc")
.agg(F.sum("spend_needed_to_redeem").alias("spend_needed_to_redeem"),
F.sum("spend_needed_ex_headroom").alias("spend_needed_ex_headroom"),
F.sum("markdown").alias("markdown"),
F.count("*").alias("count"))
.withColumn("SMR", F.col("spend_needed_to_redeem") / F.col("markdown"))
.withColumn("SMR_ex_headroom", F.col("spend_needed_ex_headroom") / F.col("markdown"))
.withColumn("avg_spend_to_red_ex_headroom", F.col("spend_needed_ex_headroom") / F.col("count"))
).display()

# COMMAND ----------



# COMMAND ----------

# MAGIC %md # SMR at individual basis 
# MAGIC - To allow to distinquish high SMR and low SMR in the same group 

# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------


extra_spend_pd = alloc_spend_gt_x.select("spend_to_markdown_ratio", "spend_needed_to_redeem", "segment_desc", "cust_band_fd").toPandas()

# COMMAND ----------



# COMMAND ----------

extra_spend_pd

# COMMAND ----------

extra_spend_pd.groupby("segment_desc").count()

# COMMAND ----------


fig = px.histogram(extra_spend_pd.sample(n = 100000), x="spend_to_markdown_ratio")
fig.show()

# COMMAND ----------

fig = px.histogram(extra_spend_pd[~(extra_spend_pd.segment_desc.isnull())].sample(n = 100000), x="spend_to_markdown_ratio", facet_row = "segment_desc")
fig.show()

# COMMAND ----------

fig = px.histogram(extra_spend_pd[~(extra_spend_pd.segment_desc.isnull()) & (extra_spend_pd.spend_to_markdown_ratio > -5)].sample(n = 100000), x="spend_to_markdown_ratio", facet_row = "segment_desc")
fig.update_yaxes(matches=None, showticklabels=True)
fig.show()

# COMMAND ----------



# COMMAND ----------



# COMMAND ----------

alloc_spend_gt_x.filter(F.col("cust_id") == -6395435731941412311).display()

# COMMAND ----------

alloc_combined.filter(F.col("cust_id") == -6395435731941412311).display()

# COMMAND ----------

alloc.filter(F.col("cust_id") == -6395435731941412311).display()

# COMMAND ----------

# MAGIC %sql select * from loyalty_azlab_prod.headroom_allocation_p_tbl where campaign = 20230426 and cust_id = '-6395435731941412311'

# COMMAND ----------

# MAGIC %md 
# MAGIC - What is the minimun spend needed to get the offer 
# MAGIC
# MAGIC - What is the markdown of the offer 
# MAGIC
# MAGIC - Ratio of minimun spend / markdown for the offer,  if its >1 its good 
# MAGIC If its < 1 means losing money 

# COMMAND ----------

# alloc_spend_gt_x = (
#   alloc_spend_gt_x
#   .withColumn("spend_needed_to_redeem", F.col("offer_spend_value") - F.col("spend_plus_headroom"))
#   .withColumn("spend_needed_ex_headroom", F.col("offer_spend_value") - F.col("estimated_spend"))
#   .withColumn("spend_to_markdown_ratio", F.col("spend_needed_to_redeem") / F.col("markdown"))
#   .withColumn("spend_to_markdown_ratio_exc_headroom", F.col("spend_needed_ex_headroom") / F.col("markdown"))
# )


# COMMAND ----------



# COMMAND ----------

# MAGIC %md ## By TCOL 

# COMMAND ----------



# COMMAND ----------

# MAGIC %sql select * from loyalty_azlab_prod.predictions_p_tbl where campaign = 20230426

# COMMAND ----------

data = spark.sql("select * from loyalty_azlab_prod.predictions_p_tbl where campaign = 20230426")

# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------

loyalty_azlab_prod.predictions_p_tbl

1

# COMMAND ----------

lab_230424 = spark.sql("select * from loyalty_azlab_prod.headroom_allocation_p_tbl where campaign = 20230424")
display(lab_230424)

# COMMAND ----------

lab_230424.groupBy('desc').count()\
  .withColumn('percentage', F.round(F.col('count') / F.sum('count')\
  .over(W.partitionBy()),3)).display()

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

lab_segment = (
  lab_230424
  .join(segtco_history_.select("cust_id", "cust_band_fd"), on = "cust_id", how = "left")
)
display(lab_segment)

# COMMAND ----------

lab_segment.groupby("cust_band_fd", "desc").count().display()

# COMMAND ----------

lab_segment.crosstab("prod_desc", "cust_band_fd").display()

# COMMAND ----------

# MAGIC %md # What is the movement between different offer levels? 

# COMMAND ----------

alloc = (
  prod_230424.select("cust_id", F.col("desc").alias("prod_desc"))
  .join(lab_230424.select("cust_id", F.col("desc").alias("lab_desc")), on = "cust_id", how = "outer")
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

capped_df = (
  lab_segment
  .withColumn("estimated_spend_capped", F.when(F.col("estimated_spend") > 200, 200).otherwise(F.col("estimated_spend")))
  .withColumn("estimated_headroom_capped", F.when(F.col("estimated_headroom") > 200, 200).otherwise(F.col("estimated_headroom")))
)
capped_df.display()


# COMMAND ----------

capped_pd = capped_df.select("desc", "cust_band_fd", "estimated_spend_capped", "estimated_headroom_capped").toPandas()

# COMMAND ----------

capped_pd

# COMMAND ----------

sns.kdeplot(data=capped_pd, x="estimated_spend_capped", y="estimated_headroom_capped", hue="desc")

# COMMAND ----------

sns.kdeplot(data=capped_pd, x="estimated_spend_capped", hue = "desc")


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
