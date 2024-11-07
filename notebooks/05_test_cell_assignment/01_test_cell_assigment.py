# Databricks notebook source
# MAGIC %run ../setup/bootstrap

# COMMAND ----------

from ast import literal_eval
from datetime import datetime, timedelta

import seaborn as sns
from dtaml.logging import get_logger
from dtaml.utils.table import factory_table

import customer_headroom.utils.persist_utils as persist_utils
from customer_headroom.test_cell_assignment import test_cell_assignment

from pyspark.sql import functions as F, DataFrame, Column, Window as W, types as T


sns.set(style="whitegrid")
logger = get_logger("customer-headroom")

# COMMAND ----------

config_al = config['allocation']
test_cells_tbl_name = persist_utils.get_table_name(
  factory_database=config.factory_database,
  lab_database=config.lab_database,
  table_prefix=config_al.full_export_tbl.prefix,
  sensitivity=config.sensitivity
)

logger.info(f"""test_cells_tbl_name: {test_cells_tbl_name}""")

test_cells_tbl = persist_utils.read_table(
    table_name=test_cells_tbl_name
)

# COMMAND ----------

full_export_selected = test_cell_assignment.random_assignment(test_cells_tbl)

# COMMAND ----------

full_export_selected_tbl_name = persist_utils.create_beam_table(
    table_prefix=config_al.full_export_tbl.prefix,
    lab_database=config.lab_database,
    factory_database=config.factory_database,
    sensitivity=config.sensitivity,
    schema=full_export_selected,
    partition_by=config_al.full_export_selected_tbl.partitionByList,
    overwrite_table=True,
    assert_equality=False,
    add_load_timestamp=True,
)
logger.info(f"""full_export_selected_tbl_name: {full_export_selected_tbl_name}""")

persist_utils.insert_df_into_table(
    target_tbl_name=full_export_selected_tbl_name,
    insert_df=full_export_selected,
    insert_append=True,
    add_columns=True,
)

# COMMAND ----------

test_cells_selected_tbl_name = persist_utils.get_table_name(
  factory_database=config.factory_database,
  lab_database=config.lab_database,
  table_prefix=config_al.full_export_selected_tbl.prefix,
  sensitivity=config.sensitivity
)

logger.info(f"""test_cells_selected_tbl_name: {test_cells_selected_tbl_name}""")

test_cells_selected_tbl = persist_utils.read_table(
    table_name=test_cells_selected_tbl_name
)
