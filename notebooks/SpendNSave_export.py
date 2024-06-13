# Databricks notebook source
dbutils.widgets.removeAll()

# COMMAND ----------

dbutils.widgets.text("sns_headroom_schema", "datascienceoffers_analyse_prod", "")
dbutils.widgets.text("sns_headroom_table", "headroom_allocation_p_tbl", "")
# datascienceoffers_analyse_prod.headroom_allocation_p_tbl
dbutils.widgets.text("sparks_account_schema", "analytics_trans_prod", "")
dbutils.widgets.text(
    "sparks_account_table", "sparks_account", ""
)  # analytics_trans_prod.sparks_account
dbutils.widgets.text(
    "customer_group_path",
    "dbfs:/mnt/centralds/offerallocation/FPO/prod_runs/220208/prod_input/tco_customer_groups_v1/customer_group=2",
    "",
)  # dbfs:/mnt/centralds/offerallocation/FPO/prod_runs/220207/prod_input/tco_customer_groups_v2/customer_group=2
dbutils.widgets.text(
    "Sns_out_path",
    "dbfs:/mnt/centralds/offerallocation/SNS/prod_runs/2208015/export/SpendNSave220815.csv",
    "",
)  # dbfs:/mnt/centralds/offerallocation/SNS/prod_runs/220801/export/SpendNSave220801.csv
dbutils.widgets.text(
    "cds_wheel_path",
    "dbfs:/FileStore/jars/offer_engine/dev/20220808.01/cdsutils-0.0.8.2021061002-py3-none-any.whl",
    "",
)
dbutils.widgets.text(
    "oe_wheel_path",
    "dbfs:/FileStore/jars/offer_engine/dev/20220808.01/offerallocation-20220801.1-py3-none-any.whl",
    "",
)

# COMMAND ----------

cds_wheel_path = dbutils.widgets.get("cds_wheel_path")
oe_wheel_path = dbutils.widgets.get("oe_wheel_path")
dbutils.library.install(cds_wheel_path)
dbutils.library.install(oe_wheel_path)
dbutils.library.restartPython()

# COMMAND ----------

import logging.config

from offerallocation.utils.logging_utils import LOGGING_CONFIG
from pyspark.sql import functions as F
from pyspark.sql import types as T

logging.config.dictConfig(LOGGING_CONFIG)
logger = logging.getLogger("offerallocation")

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

sns_headroom_table = dbutils.widgets.getArgument("sns_headroom_table")
sns_headroom_schema = dbutils.widgets.getArgument("sns_headroom_schema")
sparks_account_table = dbutils.widgets.getArgument("sparks_account_table")
sparks_account_schema = dbutils.widgets.getArgument("sparks_account_schema")
customer_group_path = dbutils.widgets.getArgument("customer_group_path")
Sns_out_path = dbutils.widgets.getArgument("Sns_out_path")
value_map = {
    "£3": 3,
    "£5": 5,
    "£7": 7,
    "£9": 9,
    "£10": 10,
    "£12": 12,
    "£14": 14,
    "£16": 16,
    "£20": 20,
}


logger.info(
    f"""
INPUT
sns_headroom_table={sns_headroom_table}
sparks_account_table={sparks_account_table}
customer_group_path={customer_group_path}
sns_headroom_schema={sns_headroom_schema}
sparks_account_schema={sparks_account_schema}

OUTPUT
Sns_out_path={Sns_out_path}
"""
)

# COMMAND ----------


def get_offer_value(strng: str) -> float:
    value = strng.split(" ")[0]
    return float(value_map[value])


get_offer_value_udf = F.udf(get_offer_value, T.FloatType())

# COMMAND ----------

offer_details = spark.sql(
    f"""select DISTINCT(offer_id),desc from {sns_headroom_schema+"."+sns_headroom_table}"""
)

# COMMAND ----------

offer_details_new = offer_details.withColumn(
    "offer_value", get_offer_value_udf(F.col("desc"))
)

# COMMAND ----------

display(offer_details_new)

# COMMAND ----------

sns_current = spark.sql(
    f"""select p.account_id,c.offer_id from {sparks_account_schema+"."+sparks_account_table} as p INNER JOIN {sns_headroom_schema+"."+sns_headroom_table} as c ON p.cust_id=c.cust_id"""
)

# COMMAND ----------

customer_set = spark.read.parquet(customer_group_path)

# COMMAND ----------

sns_pre_final = customer_set.join(sns_current, on="account_id", how="inner")

# COMMAND ----------

logger.info(
    f""" Total Unique Customer Count:{sns_pre_final.select('account_id').count()}"""
)

# COMMAND ----------

sns_final = sns_pre_final.join(
    offer_details_new.select("offer_id", "offer_value"), how="left", on="offer_id"
).withColumnRenamed("offer_id", "Offer_1")

# COMMAND ----------

sns_final.coalesce(1).write.csv(Sns_out_path, header=True, mode="overwrite")

# COMMAND ----------

df_out_full_read = spark.read.csv(Sns_out_path, header=True)

# COMMAND ----------

display(df_out_full_read)
