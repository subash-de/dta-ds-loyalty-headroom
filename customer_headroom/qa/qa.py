from pyspark.sql import Window, functions as F
from pyspark.sql import DataFrame


def random_pick_with_scope(input_df, cust_seg_col, rows_per_group=3):
    """
    Randomly picks a specified number of rows per group based on a random order, cleans the scope column,
    and returns a list of distinct cleaned scope values.

    Args:
        input_df (DataFrame): Input PySpark DataFrame.
        cust_seg_col (str): Column name representing customer segments for grouping.
        rows_per_group (int): Number of rows to select per group (default is 3).

    Returns:
        list: List of distinct cleaned scope values.
        DataFrame: PySpark DataFrame containing the randomly selected rows.
    """

     # Define the window specification
    window_spec = Window.partitionBy(["scope", cust_seg_col]).orderBy(F.rand())

    # Filter and assign row numbers within each group
    filtered_df = (
        input_df.filter(F.col(cust_seg_col).isin(["1. Top", "2. Core", "3. Occ"]))
                .withColumn("row_number", F.row_number().over(window_spec))
    )

    # Pick the top rows per group and clean the scope column
    random_picks = (
        filtered_df.filter(F.col("row_number") <= rows_per_group)
                   .drop("row_number")
                   .withColumn(
                       "cleaned_scope",
                       F.expr("substring(scope, 1, length(scope) - instr(reverse(scope), '_'))")
                   )
    )

    # Get distinct cleaned scope values and return as a list
    cleaned_scope_list = random_picks.select("cleaned_scope").distinct().toPandas()["cleaned_scope"].tolist()

    return cleaned_scope_list, random_picks
