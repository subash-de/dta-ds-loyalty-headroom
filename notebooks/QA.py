# Databricks notebook source
# MAGIC %md # QA notebook 

# COMMAND ----------

# MAGIC %run ./bootstrap 

# COMMAND ----------

from offerallocationv2.utils import tmo_utils
import pandas as pd 
import plotly
import plotly.express as px
from pyspark.sql import functions as F, DataFrame, Column, Window as W, types as T
import seaborn as sns


# COMMAND ----------

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
last_registration_date = int((datetime.strptime(str(campaign), date_format) -
                                  timedelta(days=config_dates['lookback_days_registration'])).strftime(date_format))

campaign = 20230525

logger.info(f"""
config_dates: {config_dates}
campaign: {campaign}
last_registration_date: {last_registration_date}
""")

# COMMAND ----------

# MAGIC %md # Allocation 

# COMMAND ----------

if "allocate" in config.steps:
    config_al = config["allocation"]
    headroom_tbl_name = persist_utils.get_table_name(factory_database=config_al.headroom_export_tbl.factory_database,
                                                     lab_database=config.dev_database,
                                                     table_prefix=config_al.headroom_export_tbl.prefix,
                                                     sensitivity=config_al.headroom_export_tbl.sensitivity)
    logger.info(f"""headroom_tbl_name: {headroom_tbl_name}""")

    headroom_tbl = persist_utils.read_table(table_name=headroom_tbl_name, where=f"campaign={campaign}")
    display(headroom_tbl.orderBy(F.rand()))

# COMMAND ----------

headroom_tbl.count()

# COMMAND ----------

# MAGIC %md # TCOL split 

# COMMAND ----------

# segtco_history_df = spark.table("customer_azbase_prod.segtco_history")
# segtco_history_ = tmo_utils.get_preceding_segtco_history(segtco_history_df, 20230426)
# segtco_history_.write.parquet("/mnt/centralds/offerallocation/headroom/analysis/230426/tcol_segmentation_230426")

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



# COMMAND ----------

# MAGIC %md # VIP list 

# COMMAND ----------

vip = spark.read.csv("dbfs:/mnt/centralds/offerallocation/TMO/vip_customers/vip_list_20221201", header=True)

sparks = spark.sql("select account_id, cust_id from analytics_trans_prod.sparks_account")
display(vip.join(sparks, how = "left", on = "account_id"))

# COMMAND ----------

display(alloc.join(vip.join(sparks, how = "left", on = "account_id"), how = "inner", on = "cust_id"))

# COMMAND ----------


