# Databricks notebook source
dbutils.widgets.removeAll()

# COMMAND ----------

dbutils.widget.text("sns_headroom_table","","") #datascienceoffers_analyse_prod.headroom_allocation_p_tbl
dbutils.widget.text("sparks_account_table","","") #analytics_trans_prod.sparks_account
dbutils.widget.text("customer_group_path","","")# dbfs:/mnt/centralds/offerallocation/FPO/prod_runs/220207/prod_input/tco_customer_groups_v2/customer_group=2
dbutils.widget.text("out_path","","")# dbfs:/mnt/centralds/offerallocation/SNS/prod_runs/220801/export/SpendNSave220801.csv


# COMMAND ----------

cds_wheel_path = dbutils.widgets.get("cds_wheel_path")
oe_wheel_path = dbutils.widgets.get("oe_wheel_path")
dbutils.library.install(cds_wheel_path)
dbutils.library.install(oe_wheel_path)
dbutils.library.restartPython()

# COMMAND ----------

from pyspark.sql import DataFrame, functions as F, types as T, Window as W
import pandas as pd
import logging.config

logging.config.dictConfig(LOGGING_CONFIG)
logger = logging.getLogger("offerallocation")

# COMMAND ----------

try:
    url = []
    run_data = json.loads(
        dbutils.notebook.entry_point.getDbutils().notebook().getContext().toJson()
    )
    run_id = run_data['currentRunId']['id']
    url.append(f'https://northeurope.azuredatabricks.net/api/2.0/jobs/runs/export?run_id={run_id}')
    logger.info(f"Current URL: {url}")
except:
    logger.info("This is an interactive run")

# COMMAND ----------

sns_headroom_table=dbutils.widgets.getArgument("sns_headroom_table")
sparks_account_table=dbutils.widgets.getArgument("sparks_account_table")
customer_group_path=dbutils.widgets.getArgument("customer_group_path")
out_path=dbutils.widgets.getArgument("out_path")


logger.info(f"""
INPUT
sns_headroom_table={sns_headroom_table}
sparks_account_table={sparks_account_table}
customer_group_path={customer_group_path}


OUTPUT
out_path={out_path}
)

# COMMAND ----------

offer_list=spark.sql(f"select DISTINCT(offer_id),desc from {sns_headroom_table}").toPandas()

# COMMAND ----------

offer_list.to_dict()

# COMMAND ----------

value={}
for i in range(offer_list.shape[0]):
  value[offer_list.iloc[i].offer_id]=offer_list.iloc[i].desc.split(" ")[0][1]

# COMMAND ----------

value

# COMMAND ----------

sns_current=spark.sql(f"select p.account_id,c.offer_id from {sparks_account_table} as p INNER JOIN {sns_headroom_table} as c ON p.cust_id=c.cust_id").toPandas()

# COMMAND ----------

customer_set=spark.read.parquet(customer_group_path).toPandas()

# COMMAND ----------

sns_final=pd.merge(sns_current,customer_set,on='account_id')
sns_final.shape

# COMMAND ----------

logger.info(f""" Total Unique Customer Count:{len(sns_final.account_id.unique())}""")

# COMMAND ----------

#sns_final[sns_final['offer_id']==16388]

# COMMAND ----------

#sns_final.replace(16388,14140,inplace=True)

# COMMAND ----------

#sns_final[sns_final['offer_id']==16388]

# COMMAND ----------

sns_final['value']=0

# COMMAND ----------

for i in range(sns_final.shape[0]):
  if(sns_final.at[i,'offer_id'] in value.keys()):
    sns_final.at[i,'value']=value[sns_final.at[i,'offer_id']]

# COMMAND ----------

sns_final.value=sns_final.value.astype('int32')

# COMMAND ----------

sns_final['offer']

# COMMAND ----------

sns_final.to_csv(out_path,header=True,index=False)

# COMMAND ----------

display(spark.read.csv(out_path,header=True))

# COMMAND ----------


