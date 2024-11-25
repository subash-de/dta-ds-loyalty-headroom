from pyspark.sql import functions as F


def category_headroom_upper_lim_excl(offer_variants_df, headroom_df, lx_key):
    """Figures out the upper limit for an offer and excludes customers that have a predicted headroom of more than the upper limit for each category

    Parameters
    ----------
    offer_variants_df : pyspark.dataframe, required
        the 'moot' i.e. the table describing the thresholds of each offer variant
    headroom_df: pyspark.dataframe, required
        the allocated offer for each customer category pair
    lx_key: str, required
        the hierarchy level that the headroom is being calculated at

    Returns
    -------
    headroom_filtered_df
        a dataframe with the headroom filtered for each category based on if their predicted headroom exceeds the max threshold
    """

    if "id" not in lx_key:
        lx_key = lx_key + "_id"

    offer_variants_df = (
        offer_variants_df.withColumn(
            "offer_limits_int",
            F.split(F.regexp_replace(F.col("offer_limits"), "[\\[\\]]", ""), ",").cast(
                "array<int>"
            ),
        )
        .withColumn("upper_limit", F.expr("offer_limits_int[1]"))
        .drop("offer_limits_int")
    )

    max_upper_limit = (
        offer_variants_df.select(*["Target", "upper_limit"])
        .groupby("Target")
        .agg(F.max("upper_limit").alias("max_upper_limit"))
        .withColumnRenamed("Target", lx_key)
    )

    headroom_filtered_df = (
        headroom_df.join(max_upper_limit, on=lx_key, how="inner")
        .filter(F.col("spend_plus_stretch") < F.col("max_upper_limit"))
        .drop("max_upper_limit")
    )

    return headroom_filtered_df
