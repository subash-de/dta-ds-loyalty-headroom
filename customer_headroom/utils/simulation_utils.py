import pandas as pd
from pyspark.sql import functions as F, DataFrame
from pyspark.sql.functions import col, weekofyear, to_date, expr, datediff, percent_rank
from pyspark.sql.window import Window
import matplotlib.pyplot as plt
import matplotlib.cm as cm 
import numpy as np
from typing import Union, List
from pyspark.sql.functions import lit


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
