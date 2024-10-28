import pandas as pd
from pyspark.sql import functions as F, DataFrame
from pyspark.sql.functions import col, weekofyear, to_date, expr, datediff, percent_rank
from pyspark.sql.window import Window
import matplotlib.pyplot as plt
import matplotlib.cm as cm 
import numpy as np
from typing import Union, List
from pyspark.sql.functions import lit


def add_week_number(df: DataFrame, date_column: str, start_date: str) -> DataFrame:
    """
    Adds a 'week_number' column to the DataFrame based on the difference between the date_column and the start_date.
    
    Args:
        df (DataFrame): Input DataFrame containing the date column.
        date_column (str): Name of the column containing the date.
        start_date (str): Starting date in 'yyyy-MM-dd' format.
        
    Returns:
        DataFrame: DataFrame with a new 'week_number' column.
    """

    df = df.withColumn("date_2", to_date(col(date_column).cast("string"), "yyyyMMdd"))
    df = df.withColumn("week_number",
                    (datediff(col("date_2"), to_date(lit(str(start_date)), "yyyyMMdd")) / 7).cast("int") + 1
                    )
    return df


def calculate_weekly_rolling_sum(df: DataFrame,
                                 rolling_window: int,
                                 rolling_window_col: str,
                                 date_column: str,
                                 start_date: str,
                                 column_to_sum: str,
                                 grouping_columns: Union[List[str], str]) -> DataFrame:
    """
    Calculates the weekly rolling sum of a specified column
    
    Args:
        df (DataFrame): Input DataFrame containing the data.
        rolling_window (int): The number of weeks to include in the rolling window.
        rolling_window_col (str): Name of the column to store the rolling sum in.
        date_column (str): Name of the date column.
        start_date (str): The start date to calculate week numbers from.
        column_to_sum (str): Column to calculate the rolling sum for.
        grouping_columns (Union[List[str], str]): Columns to group the data by.
        
    Returns:
        DataFrame: DataFrame with rolling weekly sums for the specified column.
    """
    df = add_week_number(df, date_column, start_date)

    if not isinstance(grouping_columns, list):
        grouping_columns = [grouping_columns]
    group_columns_with_week = grouping_columns + ["week_number"]
    weekly_df = df.groupBy(*group_columns_with_week).agg(F.sum(column_to_sum).alias(column_to_sum))

    # Filling 0s for missing weeks
    distinct_columns = {}
    for col in group_columns_with_week:
        distinct_columns[col] = weekly_df.select(col).distinct()
    cross_joined = distinct_columns[group_columns_with_week[0]]
    for col in group_columns_with_week[1:]:
        cross_joined = cross_joined.crossJoin(distinct_columns[col])
    weekly_df = cross_joined.join(weekly_df, on=group_columns_with_week, how="left")
    weekly_df = weekly_df.fillna({column_to_sum: 0})

    # Finding rolling sum
    w = (
        Window()
        .partitionBy(*grouping_columns)
        .orderBy("week_number")
        .rowsBetween(-(rolling_window - 1), Window.currentRow)
    )

    weekly_df = weekly_df.withColumn(
        rolling_window_col,
        F.sum(column_to_sum).over(w),
    )

    weekly_df = weekly_df.filter(F.col("week_number") >= rolling_window)

    return weekly_df


def calculate_baselines_plus_stretch_combs(baseline_percentiles: Union[List[int], int],
                                           df: DataFrame,
                                           rolling_window_col: str,
                                           grouping_columns: Union[List[str], str],
                                           stretch_amounts: Union[List[int], int]) -> DataFrame:
    """
    Calculates baseline percentiles and stretched combinations based on stretch amounts.
    
    Args:
        baseline_percentiles (Union[List[int], int]): List of baseline percentiles to calculate.
        df (DataFrame): Input DataFrame containing the rolling window data.
        rolling_window_col (str): Name of the rolling window column.
        grouping_columns (Union[List[str], str]): Columns to group the data by.
        stretch_amounts (Union[List[int], int]): List of stretch amounts to apply.
        
    Returns:
        DataFrame: DataFrame with baseline percentiles and stretched combinations.
    """
    if not isinstance(baseline_percentiles, list):
        baseline_percentiles = [baseline_percentiles]
    agg_exprs = [F.expr(f"percentile_approx({rolling_window_col}, {p/100})").alias(f"{int(p)}th_percentile") for p in
                 baseline_percentiles]

    # Calculating mean, median, max
    agg_exprs.extend([
        F.max(rolling_window_col).alias("max_value"),
        F.mean(rolling_window_col).alias("mean_value"),
        F.expr(f"percentile_approx({rolling_window_col}, 0.5)").alias("median_value"),
    ])

    if not isinstance(grouping_columns, list):
        grouping_columns = [grouping_columns]
    baseline_df = df.groupBy(*grouping_columns).agg(*agg_exprs)

    if not isinstance(stretch_amounts, list):
        stretch_amounts = [stretch_amounts]
    for percentile in baseline_percentiles:
        percentile_col = f"{percentile}th_percentile"
        for stretch in stretch_amounts:
            stretch_factor = 1 + (stretch / 100.0)
            stretch_col = f"baseline_{percentile}_stretch_{stretch}_perc"

            baseline_df = baseline_df.withColumn(
                stretch_col,
                baseline_df[percentile_col] * stretch_factor
            )

    return baseline_df


def calculate_percentile_rank(grouped_df: DataFrame,
                              all_df: DataFrame,
                              rolling_window_col: str,
                              percentile_columns: Union[List[str], str],
                              grouping_columns: Union[List[str], str]) -> DataFrame:
    """
    Calculates percentile rank for a given column and grouping columns.
    
    Args:
        grouped_df (DataFrame): DataFrame containing columns that percentiles need to be calculated for.
        all_df (DataFrame): DataFrame with all values (not grouped).
        rolling_window_col (str): Rolling window column to calculate percentile rank.
        percentile_columns (Union[List[str], str]): Columns to calculate percentile rank for.
        grouping_columns (Union[List[str], str]): Columns to group the data by.
        
    Returns:
        DataFrame: DataFrame with calculated percentile ranks.
    """
    if not isinstance(grouping_columns, list):
        grouping_columns = [grouping_columns]
    if not isinstance(percentile_columns, list):
        percentile_columns = [percentile_columns]
    
    final_df = None
    for percentile_col in percentile_columns:
        df = grouped_df.filter(F.col(percentile_col) > 0)
        df_combined = all_df.select(*grouping_columns, rolling_window_col).unionByName(
            df.select(*grouping_columns, col(percentile_col).alias(rolling_window_col)))

        windowSpec = Window.partitionBy(*grouping_columns).orderBy(rolling_window_col)

        df_with_percentile = df_combined.withColumn("percentile_rank", percent_rank().over(windowSpec)).dropDuplicates()
        join_condition = [df_with_percentile[col] == df[col] for col in grouping_columns]
        join_condition.append(df_with_percentile[rolling_window_col] == df[percentile_col])

        df_new_percentile = df_with_percentile.join(df, join_condition, "inner").select(
            *[df_with_percentile[col] for col in grouping_columns],  
            df_with_percentile[rolling_window_col].alias(percentile_col),  
            df_with_percentile.percentile_rank.alias(f"percentile_{percentile_col}")
        )

        if final_df is None:
            final_df = df_new_percentile
        else:
            final_df = final_df.join(df_new_percentile, grouping_columns, "left")

    return final_df


def plot_percentiles_distributions(pandas_df: pd.DataFrame, columns: list, colours=None, alpha=None, bins=None):
    """
    Plots histograms for the given percentile columns with options for colours, transparency, and bins.
    
    Args:
        pandas_df (pd.DataFrame): Pandas DataFrame containing the percentile columns.
        columns (list): List of columns to plot histograms for.
        colours (list, optional): List of colours to use for the histograms.
        alpha (list, optional): Transparency levels for each column.
        bins (int or list, optional): Number of bins for the histograms.
        
    Returns:
        None: Displays the histogram plot.
    """
    if colours is None:
        cmap = cm.get_cmap('tab10')
        colours = [cmap(i) for i in range(len(columns))]

    if alpha is None:
        base_alpha = 0.8
        alpha = [max(0.1, base_alpha - 0.1 * i) for i in range(len(columns))]

    if bins is None:
        bins = 10

    if isinstance(bins, list) and len(bins) == len(columns):
        multiple_bins = True
    else:
        multiple_bins = False

    for i, (column, colour, transparency) in enumerate(zip(columns, colours, alpha)):
        current_bins = bins[i] if multiple_bins else bins
        plt.hist(pandas_df[column], color=colour, alpha=transparency, bins=current_bins, density=True, 
                 label=column, edgecolor='none')

    min_x = min([pandas_df[col].min() for col in columns])
    max_x = max([pandas_df[col].max() for col in columns])
    buffer = 0.05 * (max_x - min_x)
    plt.xlim(min_x - buffer, max_x + buffer)

    plt.grid(False)
    plt.xlabel('Values')
    plt.ylabel('Frequency')
    plt.title('Histograms of Percentile Columns')
    plt.legend()
    plt.show()
