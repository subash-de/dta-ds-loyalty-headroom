# Databricks notebook source
# MAGIC %run ../setup/bootstrap

# COMMAND ----------

from datetime import datetime, timedelta

import seaborn as sns
from dtaml.logging import get_logger
from pyspark.sql import Window as W
from pyspark.sql import functions as F
from functools import reduce
import pandas as pd
import re

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


pip install openpyxl

# COMMAND ----------

offer_variants_pandas = pd.read_excel("Food offer variants.xlsx", sheet_name='Sheet1')
offer_variants = spark.createDataFrame(offer_variants_pandas)

# COMMAND ----------

offer_variants_tbl_name= persist_utils.create_beam_table(
    table_prefix=config.tables.offer_variants_tbl.prefix,
    lab_database=config.lab_database,
    factory_database=config.factory_database,
    sensitivity=config.sensitivity,
    schema=offer_variants,
    partition_by=config.tables.offer_variants_tbl.partitionByList,
    overwrite_table=True,
    assert_equality=False,
    add_load_timestamp=True,
)
logger.info(f"""offer_variants_tbl_name: {offer_variants_tbl_name}""")

persist_utils.insert_df_into_table(
    target_tbl_name=offer_variants_tbl_name,
    insert_df=offer_variants,
    insert_append=True,
    add_columns=True,
)

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

offer_variants_tbl.display()

# COMMAND ----------

campaign_df = None
# spark.table("campaign_analyse_prod.campaign_eligibility_p_tbl")


# COMMAND ----------

logger.info("Begin Allocation")
config_al = config["allocation"]

# COMMAND ----------

if config_al["multiple_reward_level_for_test_cell"] != "None":
  test_cells_with_multiple_reward_levels = config_al["multiple_reward_level_for_test_cell"].keys()
else:
  test_cells_with_multiple_reward_levels = []

# COMMAND ----------

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


if "headroom" in test_cells_with_multiple_reward_levels:
    reward_percs = config_al["multiple_reward_level_for_test_cell"]["headroom"]
else:
    reward_percs = ["unique"]

overwrite_table_indicator = True
for reward in reward_percs:
    # allocate for spend and save
    if config_al["tcol_allocate_separately"]:
        # TCOL segment
        segtco_history_df = persist_utils.read_table(
            table_name = config.factory_tbl_segtco_history
            )
        segtco_history_ = tmo_utils.get_preceding_segtco_history(
            segtco_history_df, campaign
        )
        if config["build_dataset"]["l1_ids"][0] == "FD":
            cust_seg_col = "cust_band_fd"
        else:
            cust_seg_col = "cust_band_ch"
        predictions = predictions.join(
            segtco_history_.select("cust_id", cust_seg_col), on="cust_id", how="left"
        )
        prediction_top = predictions.filter(F.col(cust_seg_col).contains("Top"))
        prediction_not_top = predictions.filter(~F.col(cust_seg_col).contains("Top"))
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
            email_eligibility=config["eligible_customers"],
            outlier_min=config_al["outlier_min"],
            outlier_max=config_al["outlier_max"],
            max_increase=config_al["max_increase"],
            min_increase=config_al["min_increase"],
            headroom_factor=config_al["headroom_factor"],
            fill_offer=config_al["fill_offer"],  # change this for top customer
            prev_not_bought_factor=config_al["prev_not_bought_factor"],
        )
        headroom_export_top = allocation_manager_top.get(prediction_top, campaign_df=campaign_df).withColumn(
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
            email_eligibility=config["eligible_customers"],
            outlier_min=config_al["outlier_min"],
            outlier_max=config_al["outlier_max"],
            max_increase=config_al["max_increase"],
            min_increase=config_al["min_increase"],
            headroom_factor=config_al["headroom_factor"],
            fill_offer=config_al["fill_offer"],
            prev_not_bought_factor=config_al["prev_not_bought_factor"],
        )
        headroom_export_not_top = allocation_manager_not_top.get(
            prediction_not_top,
            campaign_df=campaign_df,
        ).withColumn("campaign", F.lit(campaign))
        logger.info("Allocation NOT top customer - finished")

        headroom_export = headroom_export_top.union(headroom_export_not_top)

    else:
        logger.info("Allocating all customers")
        if config["category_level"]:
            headroom_export = None
            # Allocate offers for each category
            for pred_item in list(config["predict"]["pred_items"]):
                id_to_limit_map, id_to_desc_map = Allocator.get_offer_mapping(   
                    offer_variants_tbl=offer_variants_tbl,
                    department=config["build_dataset"]["l1_ids"][0], 
                    pred_item=pred_item, 
                    reward_perc=reward)

                allocation_manager = Allocator(
                    feature_col=config_al["feature_col"],
                    offer_limits=id_to_limit_map,
                    offer_desc=id_to_desc_map,
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

                headroom_export_temp = allocation_manager.get(predictions.filter(predictions[f'{config_al["lx_key"]}_id'] == pred_item), campaign_df=campaign_df).withColumn(
                    "campaign", F.lit(campaign)
                )
                if headroom_export is None:
                    headroom_export = headroom_export_temp
                else:
                    headroom_export = headroom_export.unionByName(headroom_export_temp, allowMissingColumns=True)
        else:
            # Get full basket offer
            id_to_limit_map, id_to_desc_map = Allocator.get_offer_mapping(
                offer_variants_tbl=offer_variants_tbl,
                department=config["build_dataset"]["l1_ids"][0],
                pred_item="full_basket",
                reward_perc=reward)

            allocation_manager = Allocator(
                feature_col=config_al["feature_col"],
                offer_limits=id_to_limit_map,
                offer_desc=id_to_desc_map,
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

            headroom_export = allocation_manager.get(predictions, campaign_df=campaign_df).withColumn(
                "campaign", F.lit(campaign)
            )
    if reward != "unique":
        headroom_export = headroom_export.withColumn("test_type", F.lit(f"headroom_{str(reward)}_perc_reward"))

    headroom_export_cnt = headroom_export.count()
    logger.info(f"""headroom_export_cnt: {predictions_cnt}""")

    stg_headroom_tbl_name = persist_utils.create_beam_table(
        table_prefix=config_al.headroom_export_tbl.prefix,
        lab_database=config.lab_database,
        factory_database=config.factory_stg_database,
        sensitivity=config.sensitivity,
        schema=headroom_export,
        partition_by=config_al.headroom_export_tbl.partitionByList,
        overwrite_table=True,
        assert_equality=False,
        add_load_timestamp=True,
    )
    logger.info(f"""stg_headroom_tbl_name: {stg_headroom_tbl_name}""")

    persist_utils.insert_df_into_table(
        target_tbl_name=stg_headroom_tbl_name,
        insert_df=headroom_export,
        delete_where=f"campaign={campaign}",
        insert_append=True,
        add_columns=True,
    )

    if config_al["aggregate_level"] == "basket":
        if config["exclude_high_spend"] is not None:
            logger.info(
                f"Remove customer whos spend_plus_stretch > {config['exclude_high_spend']}"
            )
            headroom_export = headroom_export.filter(
                F.col("spend_plus_stretch") <= config["exclude_high_spend"]
            )

    if config["min_num_basket"] is not None:
        logger.info(
            f"Remove customer who have less than {config['min_num_basket']} basket"
        )
        headroom_export = headroom_export.join(
            predictions.filter(F.col("count_user_basket") >= config["min_num_basket"])
            .select("cust_id")
            .distinct(),
            how="inner",
            on="cust_id",
        )

    headroom_export_cnt = headroom_export.count()
    logger.info(f"""headroom_export_cnt: {predictions_cnt}""")

    headroom_tbl_name = persist_utils.create_beam_table(
        table_prefix=config_al.headroom_export_tbl.prefix,
        lab_database=config.lab_database,
        factory_database=config.factory_database,
        sensitivity=config.sensitivity,
        schema=headroom_export,
        partition_by=config_al.headroom_export_tbl.partitionByList,
        overwrite_table=overwrite_table_indicator,
        assert_equality=False,
        add_load_timestamp=True,
    )
    overwrite_table_indicator = False
    logger.info(f"""headroom_tbl_name: {headroom_tbl_name}""")

    persist_utils.insert_df_into_table(
        target_tbl_name=headroom_tbl_name,
        insert_df=headroom_export,
        delete_where=f"campaign={campaign}",
        insert_append=True,
        add_columns=True,
    )

# COMMAND ----------

headroom_export.groupBy('cust_id').count().select('count').distinct().display()

# COMMAND ----------

headroom_export.display()

# COMMAND ----------

if config["category_level"]:
  headroom_export.select("l3_id").distinct().display()

# COMMAND ----------

# Fixed stretch allocation

if config["fixed_stretch"]:
    config_sim = config["baseline_stretch_simulations"]
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
    fixed_stretch_pattern = r"^\d+_stretch_\d+_perc$"
    test_cells = [col for col in fixed_stretch_tbl.columns if re.match(fixed_stretch_pattern, col)]
    # For each test cell, we need to check if there are multiple reward levels. If so, we need to run the allocation for each reward level.
    if set(test_cells).issubset(test_cells_with_multiple_reward_levels):
        intersection_test_cell = test_cells
    else:
        # Add None as a placeholder for when not all test cells are specified with multiple reward levels
        intersection_test_cell = [None]

    non_intersection_test_cell = [col for col in fixed_stretch_tbl.columns if col not in intersection_test_cell]

    fixed_stretch_export_final = []
    # Iterating through test cells in intersection_test_cell
    for test_cell in intersection_test_cell:
        if test_cell is None: # test cell is None, meaning it has no multiple reward levels
            reward_percs = ["unique"]
            columns = non_intersection_test_cell
        else: # else the test cell has multiple reward levels
            reward_percs = config_al["multiple_reward_level_for_test_cell"][test_cell]
            columns = non_intersection_test_cell + [test_cell]
        
        for reward in reward_percs:
            # for reward in reward_percs
            if config["category_level"]:
                fixed_stretch_export = None

                for pred_item in list(config["predict"]["pred_items"]):
                    id_to_limit_map, id_to_desc_map = Allocator.get_offer_mapping(
                        offer_variants_tbl=offer_variants_tbl,
                        department=config["build_dataset"]["l1_ids"][0],
                        pred_item=pred_item,
                        reward_perc=reward)

                    fixed_stretch_allocation_manager = Allocator(
                            feature_col=config_al["feature_col"],
                            offer_limits=id_to_limit_map,
                            offer_desc=id_to_desc_map,
                            user_key=config_al["user_key"],
                            outlier_min=config_al["outlier_min"],
                            outlier_max=config_al["outlier_max"],
                            max_increase=config_al["max_increase"],
                            min_increase=config_al["min_increase"],
                            headroom_factor=config_al["headroom_factor"],
                            fill_offer=config_al["fill_offer"],
                            prev_not_bought_factor=config_al["prev_not_bought_factor"],
                        )

                    fixed_stretch_export_temp = fixed_stretch_allocation_manager.get(
                        predictions=fixed_stretch_tbl.filter(F.col(f'{config_al["lx_key"]}_id') == pred_item).select(columns), 
                        headroom = False, 
                        grouping_columns = config_sim["grouping_columns"],
                        campaign_df=campaign_df,
                    )
                    fixed_stretch_export_temp = fixed_stretch_export_temp.withColumn("campaign", F.lit(campaign))
                    if fixed_stretch_export is None:
                        fixed_stretch_export = fixed_stretch_export_temp
                    else:
                        fixed_stretch_export = fixed_stretch_export.unionByName(fixed_stretch_export_temp)
            else:
                # Get full basket offer
                id_to_limit_map, id_to_desc_map = Allocator.get_offer_mapping(
                    offer_variants_tbl=offer_variants_tbl,
                    department=config["build_dataset"]["l1_ids"][0], 
                    pred_item="full_basket", 
                    reward_perc=reward)

                fixed_stretch_allocation_manager = Allocator(
                        feature_col=config_al["feature_col"],
                        offer_limits=id_to_limit_map,
                        offer_desc=id_to_desc_map,
                        user_key=config_al["user_key"],
                        outlier_min=config_al["outlier_min"],
                        outlier_max=config_al["outlier_max"],
                        max_increase=config_al["max_increase"],
                        min_increase=config_al["min_increase"],
                        headroom_factor=config_al["headroom_factor"],
                        fill_offer=config_al["fill_offer"],
                        prev_not_bought_factor=config_al["prev_not_bought_factor"],
                    )

                fixed_stretch_export = fixed_stretch_allocation_manager.get(
                    predictions=fixed_stretch_tbl.select(columns), 
                    headroom = False, 
                    grouping_columns = config_sim["grouping_columns"],
                    campaign_df=campaign_df,
                )
                fixed_stretch_export = fixed_stretch_export.withColumn("campaign", F.lit(campaign))
                if reward != "unique":
                    fixed_stretch_export = fixed_stretch_export.withColumn("test_type", F.lit(str(test_cell)+"_"+ str(reward)+"_perc_reward"))
        
            fixed_stretch_export_final.append(fixed_stretch_export)

fixed_stretch_export_final = reduce(lambda df1, df2: df1.unionByName(df2), fixed_stretch_export_final)
fixed_stretch_export_final.display()

# COMMAND ----------

fixed_stretch_export_final.groupBy("test_type").count().display()

# COMMAND ----------

fixed_stretch_export_final.groupBy("cust_id").count().select("count").distinct().display()

# COMMAND ----------

# one article unit + headroom stretch allocation
if config["one_article_unit_stretch"]:
    config_sim = config['baseline_stretch_simulations']
    one_article_unit_stretch_tbl_name = persist_utils.get_table_name(
        factory_database=config.factory_database,
        lab_database=config.lab_database,
        table_prefix=config_sim.one_article_unit_stretch_tbl.prefix,
        sensitivity=config.sensitivity,

    )

    logger.info(f"""one_article_unit_stretch_tbl_name: {one_article_unit_stretch_tbl_name}""")

    one_article_unit_stretch_tbl = persist_utils.read_table(
        table_name=one_article_unit_stretch_tbl_name
    )

    one_article_unit_plus_headroom_stretch_tbl = one_article_unit_stretch_tbl.join(
        headroom_export.select(["cust_id", f'{config_al["lx_key"]}_id', "spend_plus_stretch", "test_type"]),
        on=["cust_id", f'{config_al["lx_key"]}_id'],
        how="right"
    )

    one_article_unit_plus_headroom_stretch_tbl = one_article_unit_plus_headroom_stretch_tbl.fillna(0, subset=["avg_weekly_article_count"])

    # Double check there are no NAs in the necessary columns
    assert one_article_unit_plus_headroom_stretch_tbl.filter(one_article_unit_plus_headroom_stretch_tbl["85_stretch_one_article_unit"].isNull()).count() == 0
    assert one_article_unit_plus_headroom_stretch_tbl.filter(one_article_unit_plus_headroom_stretch_tbl["avg_weekly_article_count"].isNull()).count() == 0

    # If the average weekly article count is lower than a threshold, the customer is deemed as low and they will get the one article unit stretch. Otherwise, they will be stretched based on the headroom
    one_article_unit_plus_headroom_stretch_tbl = (one_article_unit_plus_headroom_stretch_tbl
            .withColumn("baseline_plus_stretch", 
            F.when(one_article_unit_plus_headroom_stretch_tbl["avg_weekly_article_count"] < config_sim["article_threshold_for_one_article_unit_stretch"],
            one_article_unit_plus_headroom_stretch_tbl["85_stretch_one_article_unit"])
            .otherwise(one_article_unit_plus_headroom_stretch_tbl["spend_plus_stretch"]))
    )
    
    one_article_plus_headroom_stretch_export = None
    for pred_item in list(config["predict"]["pred_items"]):
        id_to_limit_map, id_to_desc_map = Allocator.get_offer_mapping(
            offer_variants_tbl=offer_variants_tbl,
            department=config["build_dataset"]["l1_ids"][0], 
            pred_item=pred_item, 
            reward_perc="unique")

        one_article_plus_headroom_stretch_allocation_manager = Allocator(
                feature_col=config_al["feature_col"],
                offer_limits=id_to_limit_map,
                offer_desc=id_to_desc_map,
                user_key=config_al["user_key"],
                outlier_min=config_al["outlier_min"],
                outlier_max=config_al["outlier_max"],
                max_increase=config_al["max_increase"],
                min_increase=config_al["min_increase"],
                headroom_factor=config_al["headroom_factor"],
                fill_offer=config_al["fill_offer"],
                prev_not_bought_factor=config_al["prev_not_bought_factor"],
            )

        one_article_plus_headroom_stretch_export_temp = (one_article_plus_headroom_stretch_allocation_manager
                .get(predictions=one_article_unit_plus_headroom_stretch_tbl
                    .filter(one_article_unit_plus_headroom_stretch_tbl[f'{config_al["lx_key"]}_id'] == pred_item)
                    .select(["cust_id", f'{config_al["lx_key"]}_id', "85th_percentile", "baseline_plus_stretch", "test_type"]), 
                    headroom = False, 
                    grouping_columns = config_sim["grouping_columns"],
                    campaign_df=campaign_df,
                )
        )

        one_article_plus_headroom_stretch_export_temp = one_article_plus_headroom_stretch_export_temp.withColumn("campaign", F.lit(campaign))
        one_article_plus_headroom_stretch_export_temp = one_article_plus_headroom_stretch_export_temp.withColumn("test_type", F.lit("one_article_unit_plus_headroom"))
        if one_article_plus_headroom_stretch_export is None:
            one_article_plus_headroom_stretch_export = one_article_plus_headroom_stretch_export_temp
        else:
            one_article_plus_headroom_stretch_export = one_article_plus_headroom_stretch_export.unionByName(one_article_plus_headroom_stretch_export_temp)
    
    one_article_plus_headroom_stretch_export.display()


# COMMAND ----------

# one article unit + fixed percentage stretch allocation
if config["one_article_unit_stretch"] & config["fixed_stretch"]:
  one_article_unit_plus_fixed_stretch_tbl = one_article_unit_stretch_tbl.join(
    fixed_stretch_export_final.select(["cust_id", f'{config_al["lx_key"]}_id', "spend_plus_stretch", "test_type"]).distinct(),
    on=["cust_id", f'{config_al["lx_key"]}_id'],
    how="right"
  )

  one_article_unit_plus_fixed_stretch_tbl = one_article_unit_plus_fixed_stretch_tbl.fillna(0, subset=["avg_weekly_article_count"])

  # Double check there are no NAs in the necessary columns
  assert one_article_unit_plus_fixed_stretch_tbl.filter(one_article_unit_plus_fixed_stretch_tbl["85_stretch_one_article_unit"].isNull()).count() == 0
  assert one_article_unit_plus_fixed_stretch_tbl.filter(one_article_unit_plus_fixed_stretch_tbl["avg_weekly_article_count"].isNull()).count() == 0

  # If the average weekly article count is lower than a threshold, the customer is deemed as low and they will get the one article unit stretch. Otherwise, they will be stretched based on the headroom
  one_article_unit_plus_fixed_stretch_tbl = (one_article_unit_plus_fixed_stretch_tbl
      .withColumn("baseline_plus_stretch", 
      F.when(one_article_unit_plus_fixed_stretch_tbl["avg_weekly_article_count"] <  config_sim["article_threshold_for_one_article_unit_stretch"], 
      one_article_unit_plus_fixed_stretch_tbl["85_stretch_one_article_unit"])
      .otherwise(one_article_unit_plus_fixed_stretch_tbl["spend_plus_stretch"]))
  )

  one_article_plus_fixed_stretch_export = None
  for pred_item in list(config["predict"]["pred_items"]):
      id_to_limit_map, id_to_desc_map = Allocator.get_offer_mapping(
          offer_variants_tbl=offer_variants_tbl,
          department=config["build_dataset"]["l1_ids"][0], 
          pred_item=pred_item, 
          reward_perc="unique")

      one_article_plus_fixed_stretch_allocation_manager = Allocator(
              feature_col=config_al["feature_col"],
              offer_limits=id_to_limit_map,
              offer_desc=id_to_desc_map,
              user_key=config_al["user_key"],
              outlier_min=config_al["outlier_min"],
              outlier_max=config_al["outlier_max"],
              max_increase=config_al["max_increase"],
              min_increase=config_al["min_increase"],
              headroom_factor=config_al["headroom_factor"],
              fill_offer=config_al["fill_offer"],
              prev_not_bought_factor=config_al["prev_not_bought_factor"],
          )

      one_article_plus_fixed_stretch_export_temp = (one_article_plus_fixed_stretch_allocation_manager.get(
          predictions=one_article_unit_plus_fixed_stretch_tbl
          .filter(one_article_unit_plus_fixed_stretch_tbl[f'{config_al["lx_key"]}_id'] == pred_item)
          .select(
                  ["cust_id", f'{config_al["lx_key"]}_id', "85th_percentile", "baseline_plus_stretch", "test_type"]
          ),
          headroom = False, 
          grouping_columns = config_sim["grouping_columns"])
      )

      one_article_plus_fixed_stretch_export_temp = one_article_plus_fixed_stretch_export_temp.withColumn("campaign", F.lit(campaign))
      one_article_plus_fixed_stretch_export_temp = one_article_plus_fixed_stretch_export_temp.withColumn("test_type", F.concat(F.lit("one_article_unit_plus_"), F.col("test_type")))
      if one_article_plus_fixed_stretch_export is None:
          one_article_plus_fixed_stretch_export = one_article_plus_fixed_stretch_export_temp
      else:
          one_article_plus_fixed_stretch_export = one_article_plus_fixed_stretch_export.unionByName(one_article_plus_fixed_stretch_export_temp)

# COMMAND ----------

# Merging in different allocation reports
if config["one_article_unit_stretch"]:
  exports_merged = (headroom_export
                    .select(one_article_plus_headroom_stretch_export.columns)
                    .unionByName(one_article_plus_headroom_stretch_export)
                    .orderBy("cust_id")
  )
else:
  exports_merged = headroom_export

if config["fixed_stretch"]:
  exports_merged = exports_merged.select(fixed_stretch_export_final.columns).unionByName(fixed_stretch_export_final).orderBy("cust_id")

  if config["one_article_unit_stretch"]:
    exports_merged = exports_merged.unionByName(one_article_plus_fixed_stretch_export).orderBy("cust_id")


# COMMAND ----------

# Removing customers with no headroom output
headroom_customers = exports_merged.filter(F.col("test_type").startswith("headroom")).select("cust_id").distinct()
all_export = exports_merged.join(headroom_customers, on="cust_id", how="inner")

# COMMAND ----------

# Standardize the allocation output table format
if config["category_level"] == True:
  all_export = all_export.withColumnRenamed(f'{config_al["lx_key"]}_id', "scope")
  all_export = all_export.withColumn("scope", 
                                     F.concat(F.col("scope"), 
                                              F.lit("_" + config["segmentation"]["l1_id"].lower())
                                            ))
else:
  all_export = all_export.withColumn("scope", F.lit("full_basket_" + config["segmentation"]["l1_id"].lower()))

all_export = all_export.withColumn("mechanic", F.lit(config["mechanic"]))

# COMMAND ----------

# Add account_id and uk_digital_id to the allocation output table
cust_id_link_tbl_name = persist_utils.get_table_name(
  factory_database=config.factory_database,
  lab_database=config.lab_database,
  table_prefix=config["build_dataset"].headroom_cust_id_link_tbl.prefix,
  sensitivity=config.sensitivity
)

logger.info(f"""cust_id_link_tbl_name: {cust_id_link_tbl_name}""")

cust_id_link_tbl = persist_utils.read_table(
    table_name=cust_id_link_tbl_name
)

all_export = all_export.join(
  cust_id_link_tbl.filter(F.col("account_id").isNotNull()).select(["cust_id", "account_id", "uk_digital_id"]), 
  on="cust_id", 
  how="inner"
)

assert all_export.filter(F.col("account_id").isNull()).count() == 0, "Not all customers have an account ID"

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

test_cells_tbl.display()

# COMMAND ----------

test_cells_tbl.groupby("cust_id").count().select('count').distinct().display()

# COMMAND ----------

test_cells_tbl.groupby(["cust_id", "scope"]).count().select('count').distinct().display()

# COMMAND ----------

print(headroom_export.select('cust_id').join(fixed_stretch_export_final.select('cust_id'), on='cust_id', how='left_anti').select('cust_id').distinct().count())
print(fixed_stretch_export_final.select('cust_id').join(headroom_export.select('cust_id'), on='cust_id', how='left_anti').select('cust_id').distinct().count())

# COMMAND ----------

print(fixed_stretch_export_final.select('cust_id').join(headroom_export.select('cust_id'), on='cust_id', how='inner').select('cust_id').distinct().count())
print(fixed_stretch_export_final.select('cust_id').distinct().count())


# COMMAND ----------

dbutils.notebook.exit(True)

# COMMAND ----------


