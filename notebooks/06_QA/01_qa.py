# Databricks notebook source
# MAGIC %md # QA notebook
# MAGIC

# COMMAND ----------

# MAGIC %run ../setup/bootstrap

# COMMAND ----------

import os
import shutil

# from offerallocationv2.utils import tmo_utils
from datetime import datetime, timedelta

import seaborn as sns
from dtaml.logging import get_logger
from pyspark.sql import Window as W
from pyspark.sql import functions as F

import customer_headroom.utils.persist_utils as persist_utils

# COMMAND ----------

sns.set(style="whitegrid")
logger = get_logger("customer-headroom")

# COMMAND ----------

try:
    url = []
    run_data = json.loads(
        dbutils.notebook.entry_point.getDbutils().notebook().getContext().toJson()
    )
    run_id = run_data["currentRunId"]["id"]
    url.append(
        f"https://northeurope.azuredatabricks.net/api/2.0/jobs/runs/export?run_id={run_id}"
    )
    logger.info(f"Current URL: {url}")
except:
    logger.info("This is an interactive run")

# COMMAND ----------


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
env = os.environ["ENVIRONMENT"]


# campaign = 20230531
campaign_type = "headroom"

logger.info(
    f"""
config.dates: {config.dates}
campaign: {campaign}
last_registration_date: {last_registration_date}
env: {env}
"""
)

# COMMAND ----------

qa_outpath = f"offerallocation/HEADROOM/qa_outputs/campaign={campaign}"
out_path = persist_utils.get_absolute_blob_path(qa_outpath, config.mail_containers)
dbutils.fs.mkdirs(out_path)
out_path = f"/dbfs{out_path}"

# COMMAND ----------

out_path

# COMMAND ----------

# MAGIC %md # Allocation

# COMMAND ----------

config_al = config["allocation"]
headroom_tbl_name = persist_utils.get_table_name(
    factory_database=config.factory_database,
    lab_database=config.lab_database,
    table_prefix=config_al.headroom_export_tbl.prefix,
    sensitivity=config.sensitivity,
)
logger.info(f"""headroom_tbl_name: {headroom_tbl_name}""")

# COMMAND ----------


config_al = config["allocation"]
headroom_tbl_name = persist_utils.get_table_name(
    factory_database=config.factory_database,
    lab_database=config.lab_database,
    table_prefix=config_al.headroom_export_tbl.prefix,
    sensitivity=config.sensitivity,
)
logger.info(f"""headroom_tbl_name: {headroom_tbl_name}""")

headroom_tbl = persist_utils.read_table(
    table_name=headroom_tbl_name, where=f"campaign={campaign}"
)
display(headroom_tbl.orderBy(F.rand()))

# COMMAND ----------

headroom_tbl.count()

# COMMAND ----------

allocation_grouped = (
    headroom_tbl.groupBy("desc")
    .count()
    .withColumn(
        "percentage", F.round(F.col("count") / F.sum("count").over(W.partitionBy()), 3)
    )
)
allocation_grouped.display()

# COMMAND ----------

allocation_grouped.toPandas().to_csv(
    os.path.join(out_path, "offer_count.csv"), header=True, mode="w", index=False
)

# COMMAND ----------

# MAGIC %md # TCOL split

# COMMAND ----------

# segtco_history_df = spark.table("customer_azbase_prod.segtco_history")
# segtco_history_ = tmo_utils.get_preceding_segtco_history(segtco_history_df, 20230426)
# segtco_history_.write.parquet("/mnt/centralds/offerallocation/headroom/analysis/230426/tcol_segmentation_230426")

# COMMAND ----------

# alloc_combined = (
#   alloc
#   .join(food_segmentation.select("cust_id", "segment_desc"), on = "cust_id", how = "left")
#   .join(segtco_history_.select("cust_id", "cust_band_fd"), on = "cust_id", how = "left")
# )
# alloc_combined.count()
# alloc_combined.display()

# COMMAND ----------

# (alloc_combined.groupBy('desc').count()\
#   .withColumn('percentage', F.round(F.col('count') / F.sum('count')\
#   .over(W.partitionBy()),3)
# )
# .withColumn("all", F.lit("all"))
# ).display()

# COMMAND ----------



# COMMAND ----------

# MAGIC %md # VIP list

# COMMAND ----------

# vip = spark.read.csv("dbfs:/mnt/centralds/offerallocation/TMO/vip_customers/vip_list_20221201", header=True)

# sparks = spark.sql("select account_id, cust_id from analytics_trans_prod.sparks_account")
# display(vip.join(sparks, how = "left", on = "account_id"))

# COMMAND ----------

# display(alloc.join(vip.join(sparks, how = "left", on = "account_id"), how = "inner", on = "cust_id"))

# COMMAND ----------

# MAGIC %md
# MAGIC # Preparing email
# MAGIC ## Output formatting

# COMMAND ----------

text = ["<br>".join(["<u>Summary</u>"])]

text.append("<br>".join(["<u>QA checks: Allocation Volume by offer</u>"]))

text.append(allocation_grouped.orderBy("desc").toPandas().to_html())

text_all = "<br><br>".join(text)
text_html = text_all.replace("'", "").replace('"', "")

displayHTML(text_html)

# COMMAND ----------

body = f"""\
    <html>
      <head></head>
      <body>
        <p>Hi All,<br>
           <br>
           Please find below the QA summary for Headroom {campaign} campaign, and in attached file the corresponding summary tables.<br>
           <br>
           Thanks & Regards,<br>
           Offer Mission Team
           <br>
           <br>
           {text_html}
        </p>
      </body>
    </html>
    """
displayHTML(body)
with open(f"{out_path}/qa_email.html", "w") as file:
    file.write(body)

# COMMAND ----------

# MAGIC %md ## Compressing saved tables in one file

# COMMAND ----------

fileformat = "tar"
for fileformat in ("zip", "tar"):
    tmp_dest = "/tmp/qa_tables"
    shutil.make_archive(
        base_name=tmp_dest,
        format=fileformat,
        root_dir=out_path,
    )
    final_dest = f"{out_path}/qa_tables.{fileformat}"
    shutil.move(f"{tmp_dest}.{fileformat}", final_dest)

filepath = out_path.replace("/mnt/", "") + "/"

# COMMAND ----------

filename = f"qa_tables.{fileformat}"
fromEmail = config.Email.from_email
toEmail = ",".join(config.Email.to_email)


importance = "normal"
subject = f"QA Headroom allocations results - {campaign}"

environment = os.environ["ENVIRONMENT"]
if environment == "dev":
    storageaccount = config.mail_storage_account.dev
elif environment == "ppd":
    storageaccount = config.mail_storage_account.ppd
elif environment == "prod":
    storageaccount = config.mail_storage_account.prod
assert storageaccount is not None, "env not equal to dev/ppd/prod"

# COMMAND ----------

if len(url) > 0:
    dbutils.notebook.exit(url)
else:
    dbutils.notebook.exit(
        {
            "toEmail": toEmail,
            "fromEmail": fromEmail,
            "importance": importance,
            "subject": subject,
            "body": body,
            "filename": filename,
            "filepath": filepath[5:],
            "storageaccount": storageaccount,
        }
    )
