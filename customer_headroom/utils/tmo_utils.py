import logging
from collections import defaultdict
from functools import reduce
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql import types as T

logger = logging.getLogger("offerallocation")


def get_preceding_segtco_history(segtco_history: DataFrame, max_date: int):
    """Get most recent segtco_history table at a given date {max_date}

    Parameters
    ----------
    segtco_history: DataFrame :

    max_date: int :


    Returns
    -------

    """
    selected_date = (
        segtco_history.filter(F.col("yyyymmdd") <= max_date)
        .select(F.max("yyyymmdd"))
        .first()[0]
    )
    segtco_history_ = segtco_history.filter(F.col("yyyymmdd") == selected_date)

    return segtco_history_


def correct_bu_id(df: DataFrame) -> DataFrame:
    """Replace the common mis-read bu_id flags or alternative bu_id flags with the string type '0X'.
    This form is consistent with how bu_id's are represented in the articles table

    Parameters
    ----------
    df: DataFrame :


    Returns
    -------

    """
    if "bu_id" in df.columns:
        bu_map = {
            "WW": "01",
            "LG": "02",
            "Lingerie": "02",
            "MW": "03",
            "KW": "04",
            "HF": "05",
            "HW": "05",
            "Home": "05",
            "CoreHome": "05",
            "HB": "07",
            "BT": "07",
            "Beauty": "07",
            "1": "01",
            "2": "02",
            "3": "03",
            "4": "04",
            "5": "05",
            "7": "07",
        }
        df = df.withColumn(
            "bu_id", F.regexp_replace(F.col("bu_id").cast(T.StringType()), "\\s+", "")
        ).replace(to_replace=bu_map, subset=["bu_id"])
    return df


def prepare_tmo_offer_details(raw_offer_details: DataFrame) -> DataFrame:
    """Function takes raw csv input offer details and returns offer details table compatible with the allocation setup tasks.
    This should also be in line with the food offer details table which is created form the moot file.
    The columns offer_id, desc and test_sets are set to be in line with the food offer details table, common alternatives
     column names are replaced. bu_id alternatives are also replace but this is a TMO specific column.

     The test_sets column is provided as a list of integers. The integers indicating which tests are applicable for
     each offer. Since this is read in as a csv, the test_sets column is likely a string. This is then cast as an
     array of integers for future functions.

    Parameters
    ----------
    raw_offer_details: DataFrame :


    Returns
    -------

    """
    for i in raw_offer_details.columns:
        i_replace = i.replace(" ", "_").lower()
        if i_replace in ("bu", "bu_id"):
            i_replace = "bu_id"
        elif i_replace in ("tibco_id", "offer_id", "offer", "tibco"):
            i_replace = "offer_id"
        elif i_replace in ("test_set", "test_sets", "test"):
            i_replace = "test_sets"
        elif i_replace in ("offer_description", "desc", "offer_desc"):
            i_replace = "desc"

        raw_offer_details = raw_offer_details.withColumnRenamed(i, i_replace)

    offer_details = correct_bu_id(raw_offer_details)

    if "test_sets" in offer_details.columns:
        if offer_details.schema["test_sets"] != T.StructField(
            "test_sets", T.ArrayType(T.IntegerType(), True), True
        ):
            offer_details = offer_details.withColumn(
                "test_sets",
                F.split(F.regexp_replace("test_sets", r"\[|\]", ""), ",").cast(
                    "array<int>"
                ),
            )

    if "bu_id" in raw_offer_details.columns:
        offer_details = (
            offer_details
            # remove discount value from offer description
            .withColumn(
                "off_cat_id",
                F.hash(
                    "bu_id", F.regexp_replace(F.lower(F.col("desc")), r"\d+|\W+", "")
                ).cast("string"),
            )
        )

    return offer_details


def merge_input_caps(
    offer_details: DataFrame,
    generated_caps: DataFrame,
    control: float = 0.0,
    missing_cap_impute: Optional[int] = None,
    lb_max_cap_override: Optional[int] = None,
    ub_max_cap_override: Optional[int] = None,
    min_cap_override: int = 0,
) -> DataFrame:
    """Function for creating Offer details. Merges the generated caps with the predefined caps provided by the raw
    offer details table. Any caps present in the raw offer details table overrides any generated caps.
    Max and minimum caps can also be imposed on the generated caps but again will not override the input caps provided
    by the raw offer details table (if present).
    param: control - Is the level of control used in the campaign which is accounted for in the offer caps.
        Assume 10%

    Parameters
    ----------
    offer_details: DataFrame :

    generated_caps: DataFrame :

    control: float :
         (Default value = 0.)
    missing_cap_impute: Optional[int] :
         (Default value = None)
    lb_max_cap_override: Optional[int] :
         (Default value = None)
    ub_max_cap_override: Optional[int] :
         (Default value = None)
    min_cap_override: int :
         (Default value = 0)

    Returns
    -------

    """
    # If Markdown cap exists use
    if "input_markdown_cap" in offer_details.columns:
        offer_details = offer_details.withColumn(
            "markdown_max_cap", F.col("input_markdown_cap").cast(T.IntegerType())
        )

    offer_details_with_cap = offer_details.withColumn(
        "cap_generated_limit", F.lit(None).cast(T.IntegerType())
    ).withColumn("min_cap_gen", F.lit(None).cast(T.IntegerType()))
    if generated_caps:
        offer_details_with_cap = offer_details_with_cap.join(
            generated_caps.select("offer_id", "n_custs", "cap_generated"),
            on="offer_id",
            how="left",
        )
    else:
        offer_details_with_cap = offer_details_with_cap.withColumn(
            "n_custs", F.lit(None).cast(T.IntegerType())
        ).withColumn("cap_generated", F.lit(None).cast(T.IntegerType()))

    if missing_cap_impute:
        offer_details_with_cap = offer_details_with_cap.withColumn(
            "cap_generated_limit",
            F.coalesce(F.col("cap_generated"), F.lit(missing_cap_impute)),
        )
    if ub_max_cap_override:
        offer_details_with_cap = offer_details_with_cap.withColumn(
            "cap_generated_limit",
            F.least(F.col("cap_generated_limit"), F.lit(ub_max_cap_override)),
        )
    if lb_max_cap_override:
        offer_details_with_cap = offer_details_with_cap.withColumn(
            "cap_generated_limit",
            F.greatest(F.col("cap_generated_limit"), F.lit(lb_max_cap_override)),
        )

    offer_details_with_cap = offer_details_with_cap.withColumn(
        "min_cap_gen", F.lit(min_cap_override)
    )

    offer_details_with_cap_control = (
        offer_details_with_cap.withColumn(
            "max_cap",
            F.ceil(
                F.coalesce(
                    F.col("input_max_cap"),
                    F.col("cap_generated_limit"),
                    F.col("cap_generated"),
                )
                * (control + 1.0)
            ).cast(T.IntegerType()),
        )
        .withColumn(
            "min_cap",
            F.floor(
                F.coalesce(F.col("input_min_cap"), F.col("min_cap_gen"))
                * (control + 1.0)
            ).cast(T.IntegerType()),
        )
        # min cap cannot be greater than max cap
        .withColumn("min_cap", F.least(F.col("min_cap"), F.col("max_cap")))
        .drop("cap_generated_limit")
    )
    return offer_details_with_cap_control


def merge_redemption_caps(
    offer_details_caps: DataFrame,
    red_cap: DataFrame,
    missing_red_cap: float = 0.0,
    offer_id_col: str = "TIBCO_ID",
    red_cap_col="predicted_redemption_cap",
) -> DataFrame:
    """This funtions joins on the defined redemption baps provided by the input red_cap table.

    Parameters
    ----------
    offer_details_caps: DataFrame :

    red_cap: DataFrame :

    missing_red_cap: float :
         (Default value = 0.)
    offer_id_col: str :
         (Default value = "TIBCO_ID")
    red_cap_col :
         (Default value = "predicted_redemption_cap")

    Returns
    -------

    """
    if red_cap is not None:
        offer_details_caps = offer_details_caps.join(
            red_cap.select(
                F.col(offer_id_col).alias("offer_id"),
                F.col(red_cap_col).cast(T.DoubleType()).alias("redemption_cap"),
            ),
            on="offer_id",
            how="left",
        ).fillna(missing_red_cap, subset=["redemption_cap"])
    else:
        offer_details_caps = offer_details_caps.withColumn(
            "redemption_cap", F.lit(missing_red_cap).cast(T.DoubleType())
        )
    return offer_details_caps


def get_null_cols(df: DataFrame) -> Dict[str, int]:
    """
    Get columns with null values in a DataFrame.

    Args:
        df (DataFrame): The DataFrame to check for null values.

    Returns:
        Dict[str, int]: A dictionary containing column names as keys and the count of null values as values.
    """
    null_series = (
        df.select(
            [
                F.count(F.when(F.isnan(c) | F.col(c).isNull(), c)).alias(c)
                for c in df.columns
            ]
        ).toPandas()
    ).T[0]

    return null_series[null_series != 0].to_dict()


def get_gender(bu_list: List[str]) -> str:
    """
    Determine the gender based on the provided business unit list.

    Args:
        bu_list: A list of business units.

    Returns:
        A string indicating the gender:
        - "Men" if "MW" is present and "WW" is not present in bu_list.
        - "Women" if "MW" is not present and "WW" is present in bu_list.
        - "Both" if both "MW" and "WW" are present in bu_list, or if none of them are present.
    """
    if "MW" in bu_list and "WW" not in bu_list:
        return "Men"
    if "MW" not in bu_list and "WW" in bu_list:
        return "Women"
    else:
        return "Both"


def offers_remaining(
    offers_count: DataFrame,
    offer_min_volume_dict: dict,
    col_name_num_variant: str = "no_of_variants",
) -> DataFrame:

    """Calculate the volume of offers that needs assigning

    Parameters
    ----------
    offers_count : DataFrame
        Dataframe containing the count of offers
    offer_min_volume_dict : dict
        {BU : min volume}  dictionary
    offers_count: DataFrame :

    offer_min_volume_dict: dict :

    col_name_num_variant: str :
         (Default value = 'no_of_variants')

    Returns
    -------
    DataFrame
        containing additional columns of expected_offer_count, and remaining_offers to allocate

    """

    remain_offer_df_list = []

    # for each BU, calculate the amount left to fill
    for bu in offer_min_volume_dict.keys():
        remain_offer_bu = (
            offers_count.filter(F.col("BU").isin([bu]))
            .filter(
                F.col("all_variant_count") < offer_min_volume_dict[bu]["min_volume"]
            )
            .withColumn(
                "expected_offer_count",
                offer_min_volume_dict[bu]["min_volume"] / F.col(col_name_num_variant),
            )
            .withColumn(
                "remaining_offers", F.col("expected_offer_count") - F.col("count")
            )
        )
        remain_offer_df_list.append(remain_offer_bu)

    # concatenate the list of dfs to one single df
    remain_offers_combined = reduce(DataFrame.unionAll, remain_offer_df_list)
    remain_offers_combined = remain_offers_combined.orderBy(F.col("count").desc())

    return remain_offers_combined


def swap_offers(
    df: DataFrame,
    to_replace: str,
    value: str,
    swap_frac: float,
    offer_column_name: str = "offer_id",
) -> DataFrame:
    """Function to swap offers in spark dataframe

    Parameters
    ----------
    df : DataFrame
        spark data frame containing offer allocations
    to_replace : str
        Value to be replaced
    value : str
        Value to be replaced with
    swap_frac : float
        Fraction of offers to be swapped with the replacement offer
    offer_column_name : str
        (Default value = "offer_id")
        Name of the offer id column

    Returns
    -------


    """
    spark = SparkSession.builder.getOrCreate()
    logger.info(f"Replacing {swap_frac} fraction of {to_replace} with {value}")
    if df.filter(F.col(offer_column_name) == to_replace).count() > 0:
        df_truncated = df.filter(F.col(offer_column_name) == to_replace)
        df_change = df_truncated.sample(fraction=swap_frac).cache()
        df_replaced = df_change.replace(to_replace=to_replace, value=value)
        df_other_offers = df.join(
            df_change, on=["account_id", offer_column_name], how="leftanti"
        )
        df_combined = df_other_offers.unionAll(df_replaced)
        current_alloc_pd = df_combined.toPandas()
        df = spark.createDataFrame(current_alloc_pd)
        assert (
            df.filter(F.col(offer_column_name) == value).count() > 0
        ), f"{to_replace} hasn't been replaced with {value}"
    else:
        logger.info(f"{to_replace} has not been allocated")
    return df
