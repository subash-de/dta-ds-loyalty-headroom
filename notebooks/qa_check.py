# Databricks notebook source

import seaborn as sns
from pyspark.sql import functions as F
from pyspark.sql import types as T

sns.set_style("darkgrid")

# COMMAND ----------

allocation_path = (
    "dbfs:/mnt/centralds/offerallocation/headroom/prod_run/export/202112/export_v3"
)

export = spark.read.csv(allocation_path, header=True, inferSchema=True)

# COMMAND ----------

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

offer_desc_short = {
    14055: "£5 off spend £30",
    13992: "£5 off spend £40",
    13993: "£5 off spend £50",
    13994: "£7 off spend £70",
    13995: "£9 off spend £90",
    13996: "£10 off spend £100",
    13997: "£12 off spend £120",
    14241: "£14 off spend £140",
    14183: "£16 off spend £160",
    14184: "£20 off spend £200",
}

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


@F.udf(T.StringType())
def get_offer_desc(offer):
    return offer_desc[offer]


@F.udf(T.StringType())
def get_offer_desc_short(offer):
    return offer_desc_short[offer]


@F.udf(T.IntegerType())
def get_offer_upper_lim(offer):
    return offer_lims[offer][1]


export_offers = (
    export.withColumn("desc", get_offer_desc(F.col("offer_id")))
    .withColumn("desc_short", get_offer_desc_short(F.col("offer_id")))
    .withColumn("upper_lim", get_offer_upper_lim(F.col("offer_id")))
)

# COMMAND ----------

export_offers.select("account_id").distinct().count()

# COMMAND ----------

# DBTITLE 1,Offer Volumes by Upper Limit
display(
    export_offers.groupby("offer_id", "desc", "desc_short", "upper_lim")
    .count()
    .orderBy("upper_lim")
)

# COMMAND ----------

# DBTITLE 1,Unique Account_id Counts
display(
    export_offers.groupby("account_id")
    .agg(
        F.count("*").alias("account_counts"),
        F.countDistinct("account_id").alias("distinct_account_counts"),
    )
    .groupby("account_counts", "distinct_account_counts")
    .count()
)

# COMMAND ----------

# DBTITLE 1,Check store Dedupe Customer is not present in allocation
recapture = spark.sql(
    "select * from ciu_azlab_dev.20211221_ns_recapture_campaign_agg_final_offer where fd_offer_eligible='Y'"
)

display(export.join(recapture, on=[export.account_id == recapture.sparks_account_id]))

# COMMAND ----------

# DBTITLE 1,VIP List
vip = spark.read.csv(
    "dbfs:/mnt/centralds/offerallocation/TMO/vip_customers/vip_list_20210928.csv",
    header=True,
    inferSchema=True,
)

export_vip = export_offers.join(vip, on="account_id")

display(export_vip.orderBy("Surname"))

# COMMAND ----------

# DBTITLE 1,TCO Split by Offer
# Read in sparks Customers
sparks_customers = spark.sql("SELECT * from analytics_trans_prod.sparks_account ")

# Read in sparks segtco
segtco_history = spark.sql("SELECT * from customer_azbase_prod.segtco_history")
segtco_history_ = segtco_history.filter(F.col("yyyymmdd") == 20211211)

sparks_tco = (
    sparks_customers.select("cust_id", "account_id")
    .join(segtco_history_, on="cust_id", how="left")
    .withColumn(
        "TCO",
        F.when(F.col("cust_band_ch") == "1. Top", "1. Top")
        .when(F.col("cust_band_ch") == "2. Core", "2. Core")
        .when(F.col("cust_band_ch") == "3. Occ", "3. Occ")
        .otherwise("7.Null"),
    )
    .select("cust_id", "account_id", "TCO")
    .distinct()
)

export_offers_tco = export_offers.join(sparks_tco, on="account_id")

display(
    export_offers_tco.groupby("offer_id", "desc_short", "upper_lim", "TCO")
    .count()
    .orderBy("upper_lim", "TCO")
)

# COMMAND ----------

display(
    export_offers_tco.groupby("offer_id", "upper_lim", "desc_short", "TCO")
    .count()
    .orderBy("upper_lim", "TCO")
)

# COMMAND ----------
