# Databricks notebook source
# MAGIC %run ../bootstrap

# COMMAND ----------

import pandas as pd
from dtaml.logging import get_logger
import customer_headroom.utils.persist_utils as persist_utils

logger = get_logger("customer-headroom")

# COMMAND ----------

pip install openpyxl

# COMMAND ----------

# Possible static tables are documented: https://confluence.marksandspencer.app/display/LNGP/Static+tables+for+MBR

static_table_path = ""
sheet_name = ""

# COMMAND ----------

static_table_pandas = pd.read_excel(static_table_path, sheet_name=sheet_name)
static_table = spark.createDataFrame(static_table_pandas)

# COMMAND ----------

static_tbl_name= persist_utils.create_beam_table(
    table_prefix=config.tables.static_tbl.prefix,
    lab_database=config.lab_database,
    factory_database=config.factory_database,
    sensitivity=config.sensitivity,
    schema=static_table,
    partition_by=config.tables.static_tbl.partitionByList,
    overwrite_table=True,
    assert_equality=False,
    add_load_timestamp=True,
)
logger.info(f"""static_tbl_name: {static_tbl_name}""")

persist_utils.insert_df_into_table(
    target_tbl_name=static_tbl_name,
    insert_df=static_table,
    insert_append=True,
    add_columns=True,
)
