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
from functools import reduce
from pyspark.sql import functions as F, DataFrame, Column, Window as W, types as T


sns.set(style="whitegrid")
logger = get_logger("customer-headroom")

# COMMAND ----------

config_al = config["allocation"]
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

config_pd = config["predict"]
config_tcs = config["test_cell_selection"]
total_number_of_customers = test_cells_tbl.select('cust_id').distinct().count()

# COMMAND ----------

test_cells_tbl = test_cells_tbl.filter(F.col("test_type").isin(config_tcs["test_cells"]))
test_cells_tbl.groupBy('cust_id').count().select('count').distinct().display()

# COMMAND ----------

if config["category_level"]:
  number_of_offers = len(config_pd["pred_items"].keys())
  number_of_customers_per_offer = int(total_number_of_customers/number_of_offers)
  full_export_selected_tabel_name = None
  # Get the sequence of offers for customer assignment: the offer with the smallest customer base first
  sequence_of_assignment = (
      test_cells_tbl.filter(
        (test_cells_tbl["test_type"] == "headroom") &
        (test_cells_tbl["estimated_spend"] > 0))
        .groupby(f"{config_pd['pred_key']}_id")
        .agg(F.countDistinct("cust_id").alias("distinct_customer_count"))
        .orderBy('distinct_customer_count', ascending=True)
        .select(f"{config_pd['pred_key']}_id")
        .rdd.flatMap(lambda x: x).collect()
  )

  for offer in sequence_of_assignment:
    logger.info(f"Assigning customers for {offer}")
    available_customers = test_cells_tbl.filter(
                                (test_cells_tbl[f"{config_pd['pred_key']}_id"] == offer) &
                                (test_cells_tbl["test_type"] == "headroom") &
                                (test_cells_tbl["estimated_spend"] > 0)
                          ).select("cust_id").distinct().orderBy(F.rand())

    assert available_customers.count() >= number_of_customers_per_offer, f"There are not enough customers for {offer}"

    # Remove customers that have already been assigned to a category
    if full_export_selected_tabel_name is not None:
      allocated_customers = persist_utils.read_table(table_name=full_export_selected_tabel_name)

      available_customers = available_customers.join(
        allocated_customers.select("cust_id").distinct(), on="cust_id", how="leftanti"
      )
    logger.info(f"Number of customers avaiable for {offer} allocation: {available_customers.count()}")
    test_cells_tbl_offer = (test_cells_tbl
                            .filter(test_cells_tbl[f"{config_pd['pred_key']}_id"] == offer)
                            .join(available_customers, on="cust_id", how="inner")
    )

    full_export_selected_offer = test_cell_assignment.assignment(
        df=test_cells_tbl_offer, 
        user_id=config_al["user_key"],
        treatment_ratio=config_tcs["treatment_ratio"],
        test_cell_split=config_tcs[config_tcs["selection_type"]],
        method=config_tcs["selection_type"],
    )

    if test_cell_assignment.qa_for_assignment(
        original_df=test_cells_tbl_offer,
        allocation_df=full_export_selected_offer,
        user_id=config_al["user_key"],
        test_cell_split=config_tcs[config_tcs["selection_type"]],
        method=config_tcs["selection_type"]
    ):
      # If we have created the table before, we don't need to overwrite it
      # Just need to append in the new result
      if full_export_selected_tabel_name is None:
        overwrite_table_indicator = True
      else:
        overwrite_table_indicator = False
      full_export_selected_tbl_name = persist_utils.create_beam_table(
          table_prefix=config_al.full_export_selected_tbl.prefix,
          lab_database=config.lab_database,
          factory_database=config.factory_database,
          sensitivity=config.sensitivity,
          schema=full_export_selected_offer,
          partition_by=config_al.full_export_selected_tbl.partitionByList,
          overwrite_table=overwrite_table_indicator,
          assert_equality=False,
          add_load_timestamp=True,
      )
      logger.info(f"""Writing test cell assignment results to {full_export_selected_tbl_name}""")
      full_export_selected_tabel_name = full_export_selected_tbl_name
      persist_utils.insert_df_into_table(
          target_tbl_name=full_export_selected_tbl_name,
          insert_df=full_export_selected_offer,
          insert_append=True,
          add_columns=True,
      )
      logger.info(f"Number of customers allocated for {offer}: {full_export_selected_offer.select(config_al['user_key']).distinct().count()}")
    else:
      logger.error("QA for test cell assignment failed")

else:
  full_export_selected = test_cell_assignment.assignment(
    df=test_cells_tbl, 
    user_id=config_al["user_key"],
    treatment_ratio=config_tcs["treatment_ratio"],
    test_cell_split=config_tcs[config_tcs["selection_type"]],
    method=config_tcs["selection_type"],
  )
  
  if test_cell_assignment.qa_for_assignment(original_df=test_cells_tbl,
                                          allocation_df=full_export_selected,
                                          user_id=config_al["user_key"],
                                          test_cell_split=config_tcs[config_tcs["selection_type"]],
                                          method=config_tcs["selection_type"]):
    full_export_selected_tbl_name = persist_utils.create_beam_table(
        table_prefix=config_al.full_export_selected_tbl.prefix,
        lab_database=config.lab_database,
        factory_database=config.factory_database,
        sensitivity=config.sensitivity,
        schema=full_export_selected,
        partition_by=config_al.full_export_selected_tbl.partitionByList,
        overwrite_table=True,
        assert_equality=False,
        add_load_timestamp=True,
    )
    logger.info(f"""Writing test cell assignment results to {full_export_selected_tbl_name}""")
    full_export_selected_tabel_name = full_export_selected_tbl_name
    persist_utils.insert_df_into_table(
        target_tbl_name=full_export_selected_tbl_name,
        insert_df=full_export_selected,
        insert_append=True,
        add_columns=True,
    )
  else:
    logger.error("QA for test cell assignment failed")



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

# COMMAND ----------

test_cells_selected_tbl.groupby("cust_id").count().select("count").distinct().display()

# COMMAND ----------

test_cells_selected_tbl.groupby("test_type").count().display()

# COMMAND ----------

test_cells_selected_tbl.groupby("test_group").count().display()

# COMMAND ----------

test_cells_selected_tbl.groupby("scope").count().display()

# COMMAND ----------

dbutils.notebook.exit(True)
