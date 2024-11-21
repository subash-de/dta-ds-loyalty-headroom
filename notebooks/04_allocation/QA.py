# Databricks notebook source
# MAGIC %run ../setup/bootstrap

# COMMAND ----------

from datetime import datetime, timedelta

import seaborn as sns
from dtaml.logging import get_logger
from pyspark.sql import Window as W
from pyspark.sql import functions as F
import pandas as pd
import matplotlib.pyplot as plt

import customer_headroom.utils.persist_utils as persist_utils

sns.set(style="whitegrid")
logger = get_logger("customer-headroom")

# COMMAND ----------

config_al = config["allocation"]

# COMMAND ----------

test_cells_tbl_name = persist_utils.get_table_name(
  factory_database=config.factory_database,
  lab_database=config.lab_database,
  table_prefix=config_al.full_export_tbl.prefix,
  sensitivity=config.sensitivity
)

logger.info(f"""test_cells_tbl_name: {test_cells_tbl_name}""")

test_cells_tbl_name_full_basket = "loyalty_azlab_prod.test_cells_allocation_val_full_basket_test_p_tbl"
test_cells_tbl_name_category = "loyalty_azlab_prod.test_cells_allocation_val_category_test_p_tbl"

test_cells_tbl_full_basket = persist_utils.read_table(
    table_name=test_cells_tbl_name_full_basket
)
test_cells_tbl_category = persist_utils.read_table(
    table_name=test_cells_tbl_name_category
)

# COMMAND ----------

for ids in list(config["predict"]["pred_items"]):
  plot_df = test_cells_tbl_category.filter(test_cells_tbl_category["l3_id"] == ids).groupby(['test_type', 'desc']).agg(F.countDistinct('cust_id').alias('customer_count')).toPandas()
  plot_df = pd.pivot_table(plot_df, values='customer_count', index='test_type', columns='desc', aggfunc='sum')
  ax = plot_df.plot(kind='bar', stacked=True, figsize=(10, 5))
  ax.legend(fontsize=6)
  ax.set_title(f"Stacked bar plot of offer allocation for {ids}")
  # plt.show()


# COMMAND ----------

for ids in list(config["predict"]["pred_items"]):
  plot_df = test_cells_tbl_category.filter(test_cells_tbl_category["l3_id"] == ids).groupby(['test_type', 'desc']).agg(F.countDistinct('cust_id').alias('customer_count')).toPandas()
  plot_df = pd.pivot_table(plot_df, values='customer_count', index='desc', columns='test_type', aggfunc='sum')
  ax = plot_df.plot(kind='bar', stacked=True, figsize=(10, 5))
  ax.legend(fontsize=6)
  ax.set_title(f"Stacked bar plot of offer allocation for {ids}")
  # plt.show()

# COMMAND ----------


plot_df = test_cells_tbl_full_basket.groupby(['test_type', 'desc']).agg(F.countDistinct('cust_id').alias('customer_count')).toPandas()
plot_df = pd.pivot_table(plot_df, values='customer_count', index='test_type', columns='desc', aggfunc='sum')
ax = plot_df.plot(kind='bar', stacked=True, figsize=(10, 5))
ax.legend(fontsize=6)
# ax.set_title(f"Stacked bar plot of offer allocation for {ids}")
# plt.show()

# COMMAND ----------

plot_df = test_cells_tbl_full_basket.groupby(['test_type', 'desc']).agg(F.countDistinct('cust_id').alias('customer_count')).toPandas()
plot_df = pd.pivot_table(plot_df, values='customer_count', index='desc', columns='test_type', aggfunc='sum')
ax = plot_df.plot(kind='bar', stacked=True, figsize=(10, 5))
ax.legend(fontsize=6)
# ax.set_title(f"Stacked bar plot of offer allocation for {ids}")

# COMMAND ----------

config_bd = config['build_dataset']

# COMMAND ----------


