# Databricks notebook source
# MAGIC %run ../bootstrap

# COMMAND ----------

from ast import literal_eval
from datetime import datetime, timedelta
import seaborn as sns
import pandas as pd
import numpy as np
import re
import os
import matplotlib.pyplot as plt

from dtaml.logging import get_logger
from dtaml.utils.table import factory_table

import customer_headroom.utils.persist_utils as persist_utils
from customer_headroom.utils.tmo_utils import get_preceding_segtco_history
from customer_headroom.etl.build_dataset import TransactionsManagerFixedStretch
from customer_headroom.etl.etl_utils import get_date
from customer_headroom.qa import qa
from functools import reduce
from pyspark.sql import functions as F, DataFrame, Column, Window, types as T

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

config_al = config["allocation"]
config_bd = config["build_dataset"]
config_qa = config["qa"]

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


logger.info(
    f"""
config.dates: {config.dates}
campaign: {campaign}
last_registration_date: {last_registration_date}
"""
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
    table_name=test_cells_selected_tbl_name,
    where=f"""
    campaign={campaign} and 
    scope LIKE "%_{config_bd['l1_ids'].lower()}" and 
    mechanic="{config['mechanic']}" and
    test_accounts=FALSE
  """
)

# COMMAND ----------
qa_outpath = f'offerallocation/{config["mechanic"].upper()}/qa_outputs/{config["dates"]["upcoming_campaign"]}/{config["columns"]["l1_ids"]}'

if(len(config['output_qa_prefix'].strip())!=0):
    qa_outpath = f"""{qa_outpath}/{config["output_qa_prefix"]}"""
out_path = persist_utils.get_absolute_blob_path(qa_outpath, config.mail_containers)
print("output path generated ",out_path)
dbutils.fs.mkdirs(out_path)
out_path = f"/dbfs{out_path}"
logger.info(f"QA checks tables output: {out_path}")

if config["write_table"]:
    if config["write_mode"] == "overwrite":
        wmode = "w"
    elif config["write_mode"] == "errorifexists":
        wmode = "x"

# COMMAND ----------

# MAGIC %md
# MAGIC # Campaign overview

# COMMAND ----------

scope_overview = (test_cells_selected_tbl
                  .groupby(["scope"])
                  .agg(F.countDistinct(config_bd["user_id"]).alias("distinct_cust_count"))
                  .orderBy(["scope"])
                  )
#todo remove persist if it didnt really improve
scope_overview.persist()
scope_overview.display()

# COMMAND ----------

test_group_overview = (test_cells_selected_tbl
                       .groupby(["scope", "test_group"])
                       .agg(F.countDistinct(config_bd["user_id"]).alias("distinct_cust_count"))
                       .orderBy(["scope", "test_group"])
                       )
test_group_overview.persist()
test_group_overview.display()

# COMMAND ----------

if config["write_table"]:
    scope_overview.toPandas().to_csv(os.path.join(out_path, "scope_overview.csv"), index=False, header=True, mode=wmode)
    test_group_overview.toPandas().to_csv(os.path.join(out_path, "test_group_overview.csv"), index=False, header=True, mode=wmode)
scope_overview.persist()
test_group_overview.unpersist()
# COMMAND ----------

# MAGIC %md
# MAGIC # Test cell breakdown

# COMMAND ----------

segtco_history_tbl = persist_utils.read_table(
    table_name = config.factory_tbl_segtco_history
)
segtco_history = get_preceding_segtco_history(segtco_history=segtco_history_tbl,
                                              max_date=campaign)

# COMMAND ----------

if config_bd["l1_ids"] == "FD":
    cust_seg_col = "cust_band_fd"
else:
    cust_seg_col = "cust_band_ch"
test_cells_selected_tbl_with_cust_seg = test_cells_selected_tbl.join(
    segtco_history.select([config_bd["user_id"], cust_seg_col]).distinct(),
    on=config_bd["user_id"],
    how="left",
)

# COMMAND ----------

test_cells_selected_tbl_with_cust_seg = test_cells_selected_tbl_with_cust_seg.withColumn(
    "spending_threshold",
    F.regexp_extract("desc", r"(?:.*?£\d+.*?£)(\d+(\.\d+)?)", 1)
)

test_cells_selected_tbl_with_cust_seg = test_cells_selected_tbl_with_cust_seg.withColumn(
    "stretch_perc",
    F.col("estimated_stretch")/F.col("estimated_spend") * 100
)

test_cells_selected_tbl_with_cust_seg = test_cells_selected_tbl_with_cust_seg.withColumn(
    "spending_threshold_stretch_perc",
    (F.col("spending_threshold") - F.col("estimated_spend"))/F.col("estimated_spend") * 100
)

# COMMAND ----------

groupby_cols = ["scope", "test_type", "test_group"]

# COMMAND ----------
test_cells_selected_tbl_with_cust_seg.persist()
test_cell_breakdown = (test_cells_selected_tbl_with_cust_seg
                       .groupby(groupby_cols)
                       .agg(F.countDistinct(config_bd["user_id"]).alias("distinct_cust_count"))
                       )

# COMMAND ----------

test_cell_breakdown_cust_seg = (test_cells_selected_tbl_with_cust_seg
                                .groupby(groupby_cols)
                                .pivot(cust_seg_col)
                                .agg(F.countDistinct(config_bd["user_id"]))
                                )
new_column_names = [f"{col.lower()}_seg_cust_count" if col not in groupby_cols else col for col in test_cell_breakdown_cust_seg.columns]
test_cell_breakdown_cust_seg = test_cell_breakdown_cust_seg.toDF(*new_column_names)

test_cell_breakdown = test_cell_breakdown.join(
    test_cell_breakdown_cust_seg,
    on=groupby_cols,
    how="left"
)

# COMMAND ----------

agg_exprs = [
    F.expr(f"percentile_approx(estimated_spend, 0.25)").alias("25th_quantile_of_baseline_spending"),
    F.avg("estimated_spend").alias("average_of_baseline_spending"),
    F.expr(f"percentile_approx(estimated_spend, 0.75)").alias("75th_quantile_of_baseline_spending"),
    F.avg("estimated_stretch").alias("average_stretch_in_£"),
    F.avg("stretch_perc").alias("average_stretch_in_%"),
    F.avg("spending_threshold_stretch_perc").alias("average_stretch_by_spending_threshold_in_%"),
]

test_cell_breakdown_baseline = test_cells_selected_tbl_with_cust_seg.groupBy(groupby_cols).agg(*agg_exprs)

test_cell_breakdown = test_cell_breakdown.join(
    test_cell_breakdown_baseline,
    on=groupby_cols,
    how="left"
)

# COMMAND ----------

test_cell_breakdown = test_cell_breakdown.orderBy(groupby_cols)

# COMMAND ----------
test_cell_breakdown.persist()
test_cell_breakdown.display()

# COMMAND ----------

if config["write_table"]:
    test_cell_breakdown.toPandas().to_csv(os.path.join(out_path, "test_cell_breakdown.csv"), index=False, header=True, mode=wmode)
test_cell_breakdown.unpersist()
# COMMAND ----------

# MAGIC %md
# MAGIC # Flag customers with large percentage of stretch

# COMMAND ----------

test_cells_selected_tbl_with_cust_seg.filter(
    (F.col("stretch_perc") > config_qa["threshold_for_stretch_perc"]) &
    (F.col("estimated_spend") > config_qa["threshold_for_estimated_spend"])
).count()

# COMMAND ----------

test_cells_selected_tbl_with_cust_seg.filter(
    (F.col("stretch_perc") > config_qa["threshold_for_stretch_perc"]) &
    (F.col("estimated_spend") > config_qa["threshold_for_estimated_spend"])
).display()

# COMMAND ----------

# MAGIC %md
# MAGIC # Pick random customers to check

# COMMAND ----------

test_cells_selected_random_picks_scope, test_cells_selected_random_picks = qa.random_pick_with_scope(
    test_cells_selected_tbl_with_cust_seg,
    cust_seg_col,
    rows_per_group=config_qa["num_cust_per_group"],
)
test_cells_selected_tbl_with_cust_seg.unpersist()
# COMMAND ----------

trx_line_df = persist_utils.read_table(
    table_name = config.factory_tbl_all_transaction_line
)
articles_df = persist_utils.read_table(
    table_name = config.factory_tbl_lu_article
)
config_sim = config["baseline_stretch_simulations"]

# COMMAND ----------

updated_lx_ids = config_bd["lx_ids"]
updated_lx_ids["full_basket"] = {"l2_id": list(config_bd["l2_ids"])}

trx_manager_fixed_stretch = TransactionsManagerFixedStretch(
    baseline_percentiles=[50, 85, 95, 100],
    etl_date=get_date(config.dates.etl_date),
    lookback_days=config.dates.lookback_days,
    grouping_columns=[config_bd["user_id"], "l3_id"],
    rolling_window=config_sim["rolling_window"],
    rolling_window_col=f"rolling_{config_sim['rolling_window']}_week_sales",
    l1_ids=config_bd["l1_ids"],
    category_level=True,
    l2_ids=config_bd["l2_ids"],
    lx="l3",
    lx_ids=updated_lx_ids,
    user_key=config_bd["user_id"],
    exclude_items=literal_eval(config["exclude_items"]),
)

percentile_df = trx_manager_fixed_stretch.get(
    trx_line_df.join(test_cells_selected_random_picks.select(config_bd["user_id"]).distinct(),
                     on=config_bd["user_id"],
                     how="inner"),
    articles_df
)

history_spending = percentile_df.join(
    test_cells_selected_random_picks,
    (test_cells_selected_random_picks[config_bd["user_id"]] == percentile_df[config_bd["user_id"]]) &
    (test_cells_selected_random_picks["cleaned_scope"] == percentile_df["l3_id"]),
    how="inner")

# COMMAND ----------
history_spending.persist()
history_spending.display()

# COMMAND ----------

if config["write_table"]:
    history_spending.toPandas().to_csv(os.path.join(out_path, "history_spending_for_random_cust_check.csv"), index=False, header=True, mode=wmode)
history_spending.unpersist()
# COMMAND ----------

# MAGIC %md
# MAGIC # Offer allocation

# COMMAND ----------

offer_allocation_check = {}

# COMMAND ----------

offer_variants_tbl_name = persist_utils.get_table_name(
    factory_database=config.factory_database,
    lab_database=config.lab_database,
    table_prefix=config.tables.offer_variants_tbl.prefix,
    sensitivity=config.sensitivity
)

logger.info(f"""offer_variants_tbl_name: {offer_variants_tbl_name}""")


offer_variants_tbl = persist_utils.read_table(
    table_name=offer_variants_tbl_name
)

# COMMAND ----------

logger.info(f"""Number of distinct offers: {test_cells_selected_tbl.select("offer_id").distinct().count()}""")

test_cells_selected_tbl = test_cells_selected_tbl.withColumn(
    "cleaned_scope", F.expr("substring(scope, 1, length(scope) - instr(reverse(scope), '_'))"))
distinct_scope = list(test_cells_selected_tbl.select("cleaned_scope").distinct().toPandas()["cleaned_scope"])

# COMMAND ----------

# MAGIC %md
# MAGIC ### Check whether all possible offers are allocated

# COMMAND ----------

# Left anti join allocated offers onto available offers on matching offer_id and scope name
available_offers_not_all_allocated = (offer_variants_tbl
.filter(F.col("target").isin(distinct_scope))
.join(
    test_cells_selected_tbl,
    (offer_variants_tbl["offer_id"] == test_cells_selected_tbl["offer_id"]) &
    (offer_variants_tbl["target"] == test_cells_selected_tbl["cleaned_scope"]),
    how="left_anti"
)
)
available_offers_not_all_allocated.persist()
if available_offers_not_all_allocated.count() == 0:
    all_available_offers_are_allocated = True
    logger.info("All available offers are allocated.")
else:
    all_available_offers_are_allocated = False
    logger.error(f"Not all available offers for {list(available_offers_not_all_allocated.select('target').distinct().toPandas()['target'])} are allocated.")
    if config["write_table"]:
        available_offers_not_all_allocated.toPandas().to_csv(os.path.join(out_path, "available_offers_not_all_allocated.csv"), index=False, header=True, mode=wmode)
available_offers_not_all_allocated.unpersist()

# COMMAND ----------

offer_allocation_check["all_available_offers_are_allocated"] = all_available_offers_are_allocated

# COMMAND ----------

# MAGIC %md
# MAGIC ### Check whether there is any offer being allocated but not in the available offer pool

# COMMAND ----------

# Filter offer variants to only those in scope
offer_variants_tbl_in_scope = offer_variants_tbl.filter(F.col("target").isin(distinct_scope))
# Left anti join avialable offers onto allocated offers on matching offer_id and scope name
allocated_offers_not_all_available = (test_cells_selected_tbl
.join(
    offer_variants_tbl_in_scope,
    (test_cells_selected_tbl["offer_id"] == offer_variants_tbl_in_scope["offer_id"]) &
    (test_cells_selected_tbl["cleaned_scope"] == offer_variants_tbl_in_scope["target"]),
    how="left_anti"
)
)
allocated_offers_not_all_available.persist()
if allocated_offers_not_all_available.count() == 0:
    all_allocated_offers_are_available = True
    logger.info("All allocated offers are available.")
else:
    all_allocated_offers_are_available = False
    logger.error(f"Not all allocated offers for {list(allocated_offers_not_all_available.select('cleaned_scope').distinct().toPandas()['cleaned_scope'])} are available.")
    if config["write_table"]:
        allocated_offers_not_all_available.toPandas().to_csv(os.path.join(out_path, "allocated_offers_not_all_available.csv"), index=False, header=True, mode=wmode)
allocated_offers_not_all_available.unpersist()
# COMMAND ----------

offer_allocation_check["all_allocated_offers_are_available"] = all_allocated_offers_are_available

# COMMAND ----------

# MAGIC %md
# MAGIC ### Check whether all accounts have distinct offers

# COMMAND ----------

# Plain count and distinct count offer ids for each cust_id - scope pair
cust_offer_allocation_count = test_cells_selected_tbl.groupby([config_bd["user_id"], "scope"]).agg(
    F.countDistinct("offer_id").alias("distinct_count_offer_ids"),
    F.count("offer_id").alias("count_offer_ids"),
)

# COMMAND ----------

# Check if the plain count and distinct count are the same
cust_offer_allocation_non_distinct = cust_offer_allocation_count.filter(F.col("distinct_count_offer_ids") != F.col("count_offer_ids"))
cust_offer_allocation_non_distinct.persist()
if cust_offer_allocation_non_distinct.count() == 0:
    all_cust_have_distinct_offers = True
    logger.info("All customers have distinct offers.")
else:
    all_cust_have_distinct_offers = False
    logger.error(f"{cust_offer_allocation_non_distinct.count()} customers don't have distinct offers.")
    if config["write_table"]:
        cust_have_non_distinct_offers = test_cells_selected_tbl.join(
            cust_offer_allocation_non_distinct.select([config_bd["user_id"], "scope"]),
            on=[config_bd["user_id"], "scope"],
            how="inner"
        )
        cust_have_non_distinct_offers.toPandas().to_csv(os.path.join(out_path, "cust_have_non_distinct_offers.csv"), index=False, header=True, mode=wmode)
cust_offer_allocation_non_distinct.unpersist()
# COMMAND ----------

offer_allocation_check["all_cust_have_distinct_offers"] = all_cust_have_distinct_offers

# COMMAND ----------

# MAGIC %md
# MAGIC ### Check whether each customer receive the number of offers within a predefined range (eg. [1, 1])

# COMMAND ----------

cust_offer_allocation_count = test_cells_selected_tbl.groupby(config_bd["user_id"]).agg(
    F.countDistinct("offer_id").alias("distinct_count_offer_ids"),
    F.count("offer_id").alias("count_offer_ids"),
)

# COMMAND ----------

# Check if both the plain count and the distinct count are within the predefined range
cust_offer_allocation_distinct_count_check = cust_offer_allocation_count.filter(
    (F.col("distinct_count_offer_ids") < config_qa["min_offer_count"]) |
    (F.col("distinct_count_offer_ids") > config_qa["max_offer_count"])
)

cust_offer_allocation_count_check = cust_offer_allocation_count.filter(
    (F.col("count_offer_ids") < config_qa["min_offer_count"]) |
    (F.col("count_offer_ids") > config_qa["max_offer_count"])
)

if cust_offer_allocation_distinct_count_check.count() == 0 & cust_offer_allocation_count_check.count() == 0:
    all_cust_have_offer_count_within_range = True
    logger.info(f'All customers have offers within [{config_qa["min_offer_count"]}, {config_qa["max_offer_count"]}].')
else:
    all_cust_have_offer_count_within_range = False
    logger.error(f'{cust_offer_allocation_distinct_count_check.count()} customers have less than {config_qa["min_offer_count"]} or more than {config_qa["max_offer_count"]} distinct offers.')
    if config["write_table"]:
        cust_have_offer_count_outside_range = (
            (cust_offer_allocation_distinct_count_check.select(config_bd["user_id"]))
            .unionByName(cust_offer_allocation_count_check.select(config_bd["user_id"]))
            .dropDuplicates()
        )
        cust_have_offer_count_outside_range = (test_cells_selected_tbl.join(
            cust_have_offer_count_outside_range,
            on=config_bd["user_id"],
            how="inner"
        ))

        cust_have_offer_count_outside_range.toPandas().to_csv(os.path.join(out_path, "cust_have_offer_count_outside_range.csv"), index=False, header=True, mode=wmode)

# COMMAND ----------

offer_allocation_check["all_cust_have_offer_count_within_range"] = all_cust_have_offer_count_within_range

# COMMAND ----------

# MAGIC %md
# MAGIC ### Check whether no account belong to both treatment and control

# COMMAND ----------

treatment_cust = test_cells_selected_tbl.filter(F.col("test_group") == "treatment")
control_cust = test_cells_selected_tbl.filter(F.col("test_group") == "control").drop("cleaned_scope")

treatment_control_overlapping_cust = treatment_cust.join(control_cust, on=config_bd["user_id"], how="inner")

if treatment_control_overlapping_cust.count() == 0:
    treatment_and_control_exclusive = True
    logger.info("All customers in treatment and control groups are exclusive.")
else:
    treatment_and_control_exclusive = False
    logger.error(f"Customers in treatment and control groups for {list(treatment_control_overlapping_cust.select('cleaned_scope').distinct().toPandas()['cleaned_scope'])} are not exclusive.")
    if config["write_table"]:
        treatment_control_overlapping_cust.toPandas().to_csv(os.path.join(out_path, "treatment_control_overlapping_cust.csv"), index=False, header=True, mode=wmode)


# COMMAND ----------

offer_allocation_check["treatment_and_control_exclusive"] = treatment_and_control_exclusive

# COMMAND ----------

# MAGIC %md
# MAGIC ### Check whether no account belong to more than one test cell group

# COMMAND ----------

cust_appearance_count = test_cells_selected_tbl.groupBy([config_bd["user_id"], "scope"]).count()

max_count = cust_appearance_count.select(F.max("count")).collect()[0][0]
min_count = cust_appearance_count.select(F.min("count")).collect()[0][0]

test_cell_overlapping_cust = cust_appearance_count.filter(F.col("count") != 1)

if max_count == 1 & min_count == 1:
    test_cell_group_exclusive = True
    logger.info("All customers in test cells are exclusive.")
else:
    test_cell_group_exclusive = False
    logger.error(f"Customers in test cells for {list(test_cell_overlapping_cust.select('scope').distinct().toPandas()['scope'])} are not exclusive.")
    if config["write_table"]:
        test_cell_overlapping_cust = (test_cells_selected_tbl.join(
            test_cell_overlapping_cust,
            on=[config_bd["user_id"], "scope"],
            how="inner"
        ))
        test_cell_overlapping_cust.toPandas().to_csv(os.path.join(out_path, "test_cell_overlapping_cust.csv"), index=False, header=True, mode=wmode)


# COMMAND ----------

offer_allocation_check["test_cell_group_exclusive"] = test_cell_group_exclusive

# COMMAND ----------

# MAGIC %md
# MAGIC ## Overall

# COMMAND ----------

if (len(offer_allocation_check) == sum(offer_allocation_check.values())):
    logger.info("All checks passed.")
else:
    logger.error("Some checks failed, please check.")

# COMMAND ----------

offer_allocation_check = pd.DataFrame([offer_allocation_check]).transpose().reset_index()
offer_allocation_check.columns = ["Check description", "Result"]

offer_allocation_check["Result"] = np.where(offer_allocation_check["Result"] == True, "Passed", "Failed")

# COMMAND ----------

if config["write_table"]:
    offer_allocation_check.to_csv(os.path.join(out_path, "offer_allocation_check.csv"), index=False, header=True, mode=wmode)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Table for offer allocation distribution

# COMMAND ----------

groupby_cols = ["offer_id", "desc", "scope"]

# COMMAND ----------

offer_volume = test_cells_selected_tbl.groupby(groupby_cols).agg(F.countDistinct(config_bd["user_id"]).alias("count"))

offer_volume_breakdown = (test_cells_selected_tbl
                          .groupby(groupby_cols)
                          .pivot("test_group")
                          .agg(F.countDistinct(config_bd["user_id"]))
                          )
new_column_names = [f"count_{col}" if col not in groupby_cols else col for col in offer_volume_breakdown.columns]
offer_volume_breakdown = offer_volume_breakdown.toDF(*new_column_names)

offer_volume = offer_volume.join(
    offer_volume_breakdown,
    on=groupby_cols,
    how="left"
).orderBy("offer_id")

offer_volume = offer_volume.withColumn("percentage_control", F.col("count_control")/F.col("count"))

# COMMAND ----------

if config["write_table"]:
    offer_volume.toPandas().to_csv(os.path.join(out_path, "offer_volume.csv"), index=False, header=True, mode=wmode)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Plots for offer allocation distribution

# COMMAND ----------

for distinct_scope in test_cells_selected_tbl.select("scope").distinct().toPandas()["scope"]:
    for distinct_test_type in test_cells_selected_tbl.filter(F.col("scope") == distinct_scope).select("test_type").distinct().toPandas()["test_type"]:
        allocation_per_test_cell = (test_cells_selected_tbl
                                    .filter(
            (F.col("scope") == distinct_scope) &
            (F.col("test_type") == distinct_test_type)
        )
                                    .groupBy(["scope", "desc"])
                                    .agg(F.countDistinct(config_bd["user_id"]).alias("distinct_cust_count"))
                                    .toPandas()
                                    )
        allocation_per_test_cell["cust_perc"] = round((allocation_per_test_cell["distinct_cust_count"] / allocation_per_test_cell["distinct_cust_count"].sum()) * 100, 1)
        allocation_per_test_cell["reward"] = allocation_per_test_cell["desc"].str.extract(r'£(\d+(\.\d+)?)')[0].astype(float)
        allocation_per_test_cell["spending_threshold"] = allocation_per_test_cell["desc"].str.extract(r'(?:.*?£\d+.*?£)(\d+(\.\d+)?)')[0].astype(float)
        allocation_per_test_cell_pivot = pd.pivot(
            allocation_per_test_cell,
            index="spending_threshold",
            columns="reward",
            values="cust_perc"
        )
        plt.figure(figsize=(8, 6))  # Set figure size
        sns.heatmap(allocation_per_test_cell_pivot, annot=True, cmap="coolwarm", annot_kws={"size": 8})
        plt.title(f"Heatmap of offer allocation for {distinct_scope} {distinct_test_type}")
        plt.show()

# COMMAND -----------

