# Databricks notebook source
# MAGIC %run ../bootstrap

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

def get_date(date):
    if str(date).lower() == "today":
        date = datetime.now().strftime("%Y%m%d")
    return int(date)


def get_campaign(campaign, etl_date):
    if (campaign == "{campaign}") or (campaign == ""):
        campaign = get_date(etl_date)
    return campaign


# COMMAND ----------

config_pd = config["predict"]
config_tcs = config["test_cell_selection"]
config_bd = config["build_dataset"]
config_al = config["allocation"]

campaign = get_campaign(config.dates.upcoming_campaign, config.dates.etl_date)

# COMMAND ----------

campaign

# COMMAND ----------

test_cells_tbl_name = persist_utils.get_table_name(
  factory_database=config.factory_database,
  lab_database=config.lab_database,
  table_prefix=config_al.full_export_tbl.prefix,
  sensitivity=config.sensitivity
)

logger.info(f"""test_cells_tbl_name: {test_cells_tbl_name}""")

test_cells_tbl = persist_utils.read_table(
    table_name=test_cells_tbl_name,
    where=f"""
    campaign={campaign} and 
    scope LIKE "%_{config_bd['l1_ids'].lower()}" and 
    mechanic="{config['mechanic']}"
    """
)

test_accounts_tbl_name = persist_utils.get_table_name(
  factory_database=config.factory_database,
  lab_database=config.lab_database,
  table_prefix=config["tables"]["test_accounts_tbl"].prefix,
  sensitivity=config.sensitivity
)

logger.info(f"""test_accounts_tbl_name: {test_accounts_tbl_name}""")

test_accounts_tbl = persist_utils.read_table(
  table_name=test_accounts_tbl_name
)

# COMMAND ----------

test_cells_tbl.display()

# COMMAND ----------

test_cells_tbl

# COMMAND ----------

total_number_of_customers = test_cells_tbl.select('cust_id').distinct().count()

# COMMAND ----------

test_cells_tbl = test_cells_tbl.join(
  test_accounts_tbl,
  test_cells_tbl["uk_digital_id"] == test_accounts_tbl["WCS_ID"],
  how="left_anti"
)

# COMMAND ----------

test_cells_tbl.select("scope").distinct().display()

# COMMAND ----------

test_cells_tbl.groupBy("cust_id").count().groupBy("count").count().show()

# COMMAND ----------

# Assigning customers
number_of_offers = test_cells_tbl.select('scope').distinct().count()
number_of_offers = len(config_pd["pred_items"].keys())
number_of_customers_per_offer = int(total_number_of_customers/number_of_offers)

# Get the sequence of offers for food category customer assignment
full_basket_cols = test_cells_tbl\
    .filter(F.col('scope').startswith('full_basket'))\
    .select('scope')\
    .distinct().rdd.flatMap(lambda x: x).collect()

sequence_of_assignment_for_categories = (
    test_cells_tbl.filter(
        (F.col('test_type').startswith('headroom')) &
        (test_cells_tbl["estimated_spend"] > 0) &
        (~F.col("scope").isin(full_basket_cols))
    )
    .groupby("scope")
    .agg(F.countDistinct("cust_id").alias("distinct_customer_count"))
    .orderBy('distinct_customer_count', ascending=True)
    .select("scope")
    .rdd.flatMap(lambda x: x).collect()
)

sequence_of_assignment = sequence_of_assignment_for_categories + full_basket_cols

# COMMAND ----------

print(sequence_of_assignment)

# COMMAND ----------

past_assigned_categories = []
for offer in sequence_of_assignment:
  logger.info(f"Assigning customers for {offer}")
  logger.info(f"categories already assigned: {past_assigned_categories}")
  available_customers = test_cells_tbl.filter(
                              (test_cells_tbl["scope"] == offer) &
                              (F.col('test_type').startswith('headroom')) &
                              (test_cells_tbl["estimated_spend"] > 0)
                        ).select("cust_id").distinct().orderBy(F.rand())

  test_cell_split = (
    config_tcs[config_tcs["selection_type"]]['basket'][config_bd['l1_ids'].lower()]
    if offer.startswith('full_basket')
    else config_tcs[config_tcs["selection_type"]]['lx_id'][config_bd['l1_ids'].lower()]
  )
  if config_tcs["selection_type"] == "volume":
    assert available_customers.count() >= sum(list(test_cell_split["treatment"].values())) + sum(list(test_cell_split["control"].values())), f"There are not enough customers for {offer}"
  else:
    assert available_customers.count() >= number_of_customers_per_offer, f"There are not enough customers for {offer}"

  logger.info(f"Number of total customers that could be allocated for {offer}: {available_customers.count()}")
  # Remove customers that have already been assigned to a category

  if len(past_assigned_categories)>0:
    past_assigned_categories_sql = ','.join(map(repr, past_assigned_categories))
    allocated_customers = persist_utils.read_table(table_name=full_export_selected_table_name, where=f"campaign='{campaign}' and scope IN ({past_assigned_categories_sql}) and mechanic='{config['mechanic']}'")

    available_customers = available_customers.join(
      allocated_customers.select("cust_id").distinct(), on="cust_id", how="leftanti"
    )

  logger.info(f"Number of customers available for {offer} allocation: {available_customers.count()}")
  test_cells_tbl_offer = (test_cells_tbl
                          .filter(test_cells_tbl["scope"] == offer)
                          .join(available_customers, on="cust_id", how="inner")
  )

  test_cells_tbl_offer = test_cells_tbl_offer.filter(F.col('test_type').isin(list(test_cell_split['treatment'].keys())))

  full_export_selected_offer = test_cell_assignment.assignment(
      df=test_cells_tbl_offer,
      user_id=config_al["user_key"],
      treatment_ratio=config_tcs["treatment_ratio"],
      test_cell_split=test_cell_split,
      method=config_tcs["selection_type"],
  )

  if test_cell_assignment.qa_for_assignment(
      original_df=test_cells_tbl_offer,
      allocation_df=full_export_selected_offer,
      user_id=config_al["user_key"],
      test_cell_split=test_cell_split,
      method=config_tcs["selection_type"]
  ):
    past_assigned_categories.append(offer)

    full_export_selected_tbl_name = persist_utils.create_beam_table(
        table_prefix=config_al.full_export_selected_tbl.prefix,
        lab_database=config.lab_database,
        factory_database=config.factory_database,
        sensitivity=config.sensitivity,
        schema=full_export_selected_offer,
        partition_by=config_al.full_export_selected_tbl.partitionByList,
        overwrite_table=False,
        assert_equality=False,
        add_load_timestamp=True,
    )
    logger.info(f"""Writing test cell assignment results to {full_export_selected_tbl_name}""")
    full_export_selected_table_name = full_export_selected_tbl_name
    persist_utils.insert_df_into_table(
        target_tbl_name=full_export_selected_tbl_name,
        insert_df=full_export_selected_offer,
        insert_append=True,
        add_columns=True,
        delete_where=f"""campaign = "{campaign}" and scope = "{offer}" and mechanic = "{config['mechanic']}"
        """
    )
    logger.info(f"Number of customers allocated for {offer}: {full_export_selected_offer.select(config_al['user_key']).distinct().count()}")
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

scope_list = ','.join(map(repr, sequence_of_assignment))

test_cells_selected_tbl = persist_utils.read_table(
  table_name=test_cells_selected_tbl_name,
  where=f"campaign='{campaign}' and scope IN ({scope_list}) and mechanic='{config['mechanic']}'"
)

# COMMAND ----------

test_cells_selected_tbl.groupby("cust_id").count().select("count").distinct().display()

# COMMAND ----------

test_cells_selected_tbl.groupBy('scope','test_type', 'test_group').agg({'cust_id': 'count'}).display()

# COMMAND ----------

# allocate random offers for test accounts
num_test_accounts = test_accounts_tbl.select("WCS_ID").distinct().count()

# COMMAND ----------

window_spec = W.partitionBy("scope").orderBy(F.rand())

test_account_offer_allocation = test_cells_selected_tbl.withColumn("row_number", F.row_number().over(window_spec))

test_account_offer_allocation = test_account_offer_allocation.filter(F.col("row_number") <= num_test_accounts)

# COMMAND ----------

window_spec = W.orderBy(F.rand())

test_accounts_tbl_temp = test_accounts_tbl.withColumn("row_number", F.row_number().over(window_spec))


# COMMAND ----------

test_account_offer_allocation = test_account_offer_allocation.join(
  test_accounts_tbl_temp.select(["WCS_ID", "row_number"]),
  on="row_number",
  how="left"
).drop("row_number")

# COMMAND ----------

test_account_offer_allocation.groupBy("WCS_ID").count().select("count").distinct().display()

# COMMAND ----------

test_account_offer_allocation = (test_account_offer_allocation
                                 .drop("uk_digital_id")
                                 .withColumnRenamed("WCS_ID", "uk_digital_id")
                                 .withColumn("test_group", F.lit("treatment"))
                                 .withColumn("test_accounts", F.lit(True))
)
test_account_offer_allocation = test_account_offer_allocation.select(test_cells_selected_tbl.columns)

reference_schema = test_cells_selected_tbl.select("cust_id", "account_id", "test_type", "spend_plus_stretch", "estimated_spend", "estimated_stretch").schema
for field in reference_schema:
  test_account_offer_allocation = test_account_offer_allocation.withColumn(field.name, F.lit(None).cast(field.dataType))

# COMMAND ----------

full_export_selected_tbl_name = persist_utils.create_beam_table(
    table_prefix=config_al.full_export_selected_tbl.prefix,
    lab_database=config.lab_database,
    factory_database=config.factory_database,
    sensitivity=config.sensitivity,
    schema=test_account_offer_allocation,
    partition_by=config_al.full_export_selected_tbl.partitionByList,
    overwrite_table=False,
    assert_equality=False,
    add_load_timestamp=True,
)
logger.info(f"""Writing test accounts offer allocation results to {full_export_selected_tbl_name}""")

scope_list = ','.join(map(repr, test_account_offer_allocation.select("scope").distinct().toPandas()["scope"]))

persist_utils.insert_df_into_table(
    target_tbl_name=full_export_selected_tbl_name,
    insert_df=test_account_offer_allocation,
    insert_append=True,
    add_columns=True,
    delete_where=f"""
    campaign={campaign} and
    scope IN ({scope_list}) and
    mechanic="{config['mechanic']}" and
    test_accounts=TRUE
    """
)

# COMMAND ----------

dbutils.notebook.exit(True)
