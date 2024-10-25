# Databricks notebook source
# MAGIC %run ../setup/bootstrap

# COMMAND ----------

from datetime import datetime, timedelta

import seaborn as sns
from dtaml.logging import get_logger
from pyspark.sql import Window as W
from pyspark.sql import functions as F

import customer_headroom.utils.persist_utils as persist_utils
from customer_headroom.allocation.allocator import Allocator
from customer_headroom.utils import tmo_utils

sns.set(style="whitegrid")
logger = get_logger("customer-headroom")

# COMMAND ----------


def find_all_segments(data, partitionByList):
    segs = (
        data.select(partitionByList)
        .distinct()
        .rdd.map(lambda x: {k: v for (k, v) in zip(partitionByList, x)})
        .collect()
    )
    return segs


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


logger.info(
    f"""
config.dates: {config.dates}
campaign: {campaign}
last_registration_date: {last_registration_date}
"""
)

# COMMAND ----------


logger.info("Begin Allocation")
config_al = config["allocation"]

# Headroom allocation
under_predict_adjustment_factor = config_al["headroom_factor"]

prediction_tbl_name = persist_utils.get_table_name(
    factory_database=config.factory_database,
    lab_database=config.lab_database,
    table_prefix=config_al.prediction_tbl.prefix,
    sensitivity=config.sensitivity,
)
logger.info(f"""prediction_tbl_name: {prediction_tbl_name}""")

predictions = persist_utils.read_table(
    table_name=prediction_tbl_name, where=f"campaign={campaign}"
)

# if its under predicting, then would force the stretch to be 20%
predictions = predictions.withColumn(
    "prediction_out_orig", F.lit(F.col("prediction_out"))
).withColumn(
    "prediction_out",
    F.when(
        F.col("prediction_out_orig") < F.col(config_al["feature_col"]),
        F.col(config_al["feature_col"]) * under_predict_adjustment_factor,
    ).otherwise(F.col("prediction_out_orig")),
)
predictions_cnt = predictions.count()
logger.info(f"""predictions_cnt: {predictions_cnt}""")

# spend and save feature column l2_id_total_spend_basket

# allocate for spend and save
if config_al["tcol_allocate_separately"]:
    # TCOL segment
    segtco_history_df = spark.table("customer_azbase_prod.segtco_history")
    segtco_history_ = tmo_utils.get_preceding_segtco_history(
        segtco_history_df, campaign
    )

    predictions = predictions.join(
        segtco_history_.select("cust_id", "cust_band_fd"), on="cust_id", how="left"
    )
    prediction_top = predictions.filter(F.col("cust_band_fd").contains("Top"))
    prediction_not_top = predictions.filter(~F.col("cust_band_fd").contains("Top"))
    assert (
        prediction_top.count() > 0
    ), f"No of rows for customer in top group, got {prediction_top.count()}"

    # top allocation
    logger.info(
        f"Allocation top customer, number of top customers {prediction_top.select('cust_id').distinct().count()}"
    )
    allocation_manager_top = Allocator(
        feature_col=config_al["feature_col"],
        offer_limits=config["offer_limits_top"],  # change this for new top offer
        offer_desc=config["offers_desc_top"],  # change this for new top offer
        user_key=config_al["user_key"],
        lx_key=config_al["lx_key"],
        outlier_min=config_al["outlier_min"],
        outlier_max=config_al["outlier_max"],
        max_increase=config_al["max_increase"],
        min_increase=config_al["min_increase"],
        headroom_factor=config_al["headroom_factor"],
        fill_offer=config_al["fill_offer"],  # change this for top customer
        prev_not_bought_factor=config_al["prev_not_bought_factor"],
    )
    headroom_export_top = allocation_manager_top.get(prediction_top).withColumn(
        "campaign", F.lit(campaign)
    )
    logger.info("Allocation top customer - finished")

    # non-top allocation
    logger.info(
        f"Allocation NOT top customer, number of none top customers {prediction_not_top.select('cust_id').distinct().count()}"
    )
    allocation_manager_not_top = Allocator(
        feature_col=config_al["feature_col"],
        offer_limits=config["offer_limits"],
        offer_desc=config["offers_desc"],
        user_key=config_al["user_key"],
        outlier_min=config_al["outlier_min"],
        outlier_max=config_al["outlier_max"],
        max_increase=config_al["max_increase"],
        min_increase=config_al["min_increase"],
        headroom_factor=config_al["headroom_factor"],
        fill_offer=config_al["fill_offer"],
        prev_not_bought_factor=config_al["prev_not_bought_factor"],
    )
    headroom_export_not_top = allocation_manager_not_top.get(
        prediction_not_top
    ).withColumn("campaign", F.lit(campaign))
    logger.info("Allocation NOT top customer - finished")

    headroom_export = headroom_export_top.union(headroom_export_not_top)

else:
    logger.info("Allocating all customers")
    allocation_manager = Allocator(
        feature_col=config_al["feature_col"],
        offer_limits=config["offer_limits"],
        offer_desc=config["offers_desc"],
        user_key=config_al["user_key"],
        lx_key=config_al["lx_key"],
        outlier_min=config_al["outlier_min"],
        outlier_max=config_al["outlier_max"],
        max_increase=config_al["max_increase"],
        min_increase=config_al["min_increase"],
        headroom_factor=config_al["headroom_factor"],
        fill_offer=config_al["fill_offer"],
        prev_not_bought_factor=config_al["prev_not_bought_factor"],
        prev_not_bought_factor_lx_id_indpendent=config_al[
            "prev_not_bought_factor_lx_id_indpendent"
        ],
        aggregate_level=config_al["aggregate_level"],
    )

    headroom_export = allocation_manager.get(predictions).withColumn(
        "campaign", F.lit(campaign)
    )

    headroom_export = headroom_export.withColumn("test_type", F.lit("headroom"))

# if config_al["aggregate_level"] == "basket":
#     if config["exclude_high_spend"] is not None:
#         logger.info(
#             f"Remove customer whos spend_plus_headroom > {config['exclude_high_spend']}"
#         )
#         headroom_export = headroom_export.filter(
#             F.col("spend_plus_headroom") <= config["exclude_high_spend"]
#         )

#     if config["min_num_basket"] is not None:
#         logger.info(
#             f"Remove customer who have less than {config['min_num_basket']} basket"
#         )
#         headroom_export = headroom_export.join(
#             predictions.filter(F.col("count_user_basket") >= config["min_num_basket"])
#             .select("cust_id")
#             .distinct(),
#             how="inner",
#             on="cust_id",
#         )

headroom_export_cnt = headroom_export.count()
logger.info(f"""headroom_export_cnt: {predictions_cnt}""")

# headroom_tbl_name = persist_utils.create_beam_table(
#     table_prefix=config_al.headroom_export_tbl.prefix,
#     lab_database=config.lab_database,
#     factory_database=config.factory_database,
#     sensitivity=config.sensitivity,
#     schema=headroom_export,
#     partition_by=config_al.headroom_export_tbl.partitionByList,
#     overwrite_table=True,
#     assert_equality=False,
#     add_load_timestamp=True,
# )
# logger.info(f"""headroom_tbl_name: {headroom_tbl_name}""")

# persist_utils.insert_df_into_table(
#     target_tbl_name=headroom_tbl_name,
#     insert_df=headroom_export,
#     delete_where=f"campaign={campaign}",
#     insert_append=True,
#     add_columns=True,
# )


# COMMAND ----------

predictions.count()

# COMMAND ----------

headroom_export.count()

# COMMAND ----------

headroom_tbl.groupBy("desc").count().withColumn(
    "percentage", F.round(F.col("count") / F.sum("count").over(W.partitionBy()), 3)
).display()

# COMMAND ----------

# Fixed stretch allocation
config_sim = config['baseline_stretch_simulations']
fixed_stretch_tbl_name = persist_utils.get_table_name(
      factory_database=config.factory_database,
    lab_database=config.lab_database,
    table_prefix=config_sim.fixed_stretch_tbl.prefix,
    sensitivity=config.sensitivity,

)

logger.info(f"""fixed_stretch_tbl_name: {fixed_stretch_tbl_name}""")

fixed_stretch_tbl = persist_utils.read_table(
    table_name=fixed_stretch_tbl_name
)

fixed_stretch_allocation_manager = Allocator(
        feature_col=config_al["feature_col"],
        offer_limits=config["offer_limits"],
        offer_desc=config["offers_desc"],
        user_key=config_al["user_key"],
        outlier_min=config_al["outlier_min"],
        outlier_max=config_al["outlier_max"],
        max_increase=config_al["max_increase"],
        min_increase=config_al["min_increase"],
        headroom_factor=config_al["headroom_factor"],
        fill_offer=config_al["fill_offer"],
        prev_not_bought_factor=config_al["prev_not_bought_factor"],
    )

# Fixed stretch allocation
fixed_stretch_export = fixed_stretch_allocation_manager.get(predictions= fixed_stretch_tbl, headroom = False)

fixed_stretch_export = fixed_stretch_export.withColumn("campaign", F.lit(campaign))


# COMMAND ----------

fixed_stretch_tbl.display()

# COMMAND ----------

fixed_stretch_export.display()

# COMMAND ----------

exports_merged = headroom_export.unionByName(fixed_stretch_export).orderBy('cust_id')

# COMMAND ----------

# customers that don't have headroom
exports_merged.filter(exports_merged["test_type"] != "headroom").select('cust_id').distinct().count()

# COMMAND ----------

# customers that don't have other fixed test cells
exports_merged.filter((exports_merged["test_type"] != 'baseline_85_stretch_0_perc') & (exports_merged["test_type"] != "baseline_85_stretch_10_perc") & (exports_merged["test_type"] != "baseline_85_stretch_30_perc")).select('cust_id').distinct().count()

# COMMAND ----------

# Removing customers with no headroom output
headroom_customers = exports_merged.filter(exports_merged["test_type"] == "headroom").select("cust_id").distinct()
all_export = exports_merged.join(headroom_customers, on="cust_id", how="inner")


# COMMAND ----------

all_export.orderBy('cust_id','test_type').display()

# COMMAND ----------

fixed_stretch_export.select('test_type').distinct().show()

# COMMAND ----------

exports_merged.select('test_type').distinct().show()

# COMMAND ----------

# Assigning random test cells numbers

# Step 1: Get distinct cust_id and shuffle them randomly
distinct_customers = all_export.select("cust_id").distinct().orderBy(F.rand())

# Step 2: Count the number of distinct test types
test_types = all_export.select("test_type").distinct().collect()
num_test_types = len(test_types)

# Step 3: Assign row numbers to the distinct customers
window_spec = W.orderBy(F.lit(1))
distinct_customers = distinct_customers.withColumn("row_num", F.row_number().over(window_spec))

test_type_assignments = F.when((distinct_customers["row_num"] % num_test_types) == 0, test_types[0][0])
for i in range(1, num_test_types):
    test_type_assignments = test_type_assignments.when((distinct_customers["row_num"] % num_test_types) == i, test_types[i][0])

distinct_customers = distinct_customers.withColumn("test_type", test_type_assignments)

# Step 5: Join this back to the original dataframe to keep only one row per customer with the assigned test_type
final_df = distinct_customers.join(all_export, on=["cust_id", "test_type"], how="inner").dropDuplicates(["cust_id"])

# Step 6: Show the result
final_df.display()



# COMMAND ----------

test_types

# COMMAND ----------

final_df.select('test_type').distinct().show()

# COMMAND ----------

all_export_with_test_type.count()

# COMMAND ----------

test_cells_tbl_name = persist_utils.create_beam_table(
    table_prefix=config_al.full_export_tbl.prefix,
    lab_database=config.lab_database,
    factory_database=config.factory_database,
    sensitivity=config.sensitivity,
    schema=all_export,
    partition_by=config_al.full_export_tbl.partitionByList,
    overwrite_table=True,
    assert_equality=False,
    add_load_timestamp=True,
)
logger.info(f"""test_cells_tbl_name: {test_cells_tbl_name}""")

persist_utils.insert_df_into_table(
    target_tbl_name=test_cells_tbl_name,
    insert_df=all_export,
    insert_append=True,
    add_columns=True,
)


# COMMAND ----------

dbutils.notebook.exit(True)
