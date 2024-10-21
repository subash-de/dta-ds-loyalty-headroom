import pandas as pd
from pyspark.sql import functions as F
from pyspark.sql.functions import col, weekofyear, to_date, expr, datediff, percent_rank
from pyspark.sql.window import Window
import matplotlib.pyplot as plt


def calculate_weekly_rolling_sum(df,
                                 rolling_window,
                                 date_column: str,
                                 start_date: str,
                                 column_to_sum,
                                 group_columns: list):
    df = add_week_number(df, date_column, start_date)

    group_columns_with_week = group_columns + ["week_number"]
    weekly_df = df.groupBy(*group_columns_with_week).agg(F.sum(column_to_sum))

    # Filling non-spending weeks with 0
    distinct_columns = {}
    for col in group_columns_with_week:
        distinct_columns[col] = weekly_df.select(col).distinct()

    cross_joined = distinct_columns[group_columns_with_week[0]]
    for col in group_columns_with_week[1:]:
        cross_joined = cross_joined.crossJoin(distinct_columns[col])
    weekly_df = cross_joined.join(weekly_df, on=group_columns, how="left")
    weekly_df = weekly_df.fillna({column_to_sum: 0})

    # Finding rolling sum
    w = (
        Window()
        .partitionBy(*group_columns)
        .orderBy("week_number")
        .rowsBetween(-(rolling_window - 1), Window.currentRow)
    )

    weekly_df = weekly_df.withColumn(
        f"rolling_{rolling_window}_week_sales",
        F.sum(column_to_sum).over(w),
    )

    weekly_df = weekly_df.filter(F.col("week_number") >= rolling_window)

    return weekly_df

def calculate_baselines_plus_stretch_combs(baseline_percentiles: list,
                                     df,
                                     rolling_window_col,
                                     grouping_columns,
                                     stretch_amounts: list):

    # Calculating baselines
    agg_exprs = [F.expr(f"percentile_approx({rolling_window_col}, {p})").alias(f"{int(p)}th_percentile") for p in
                 baseline_percentiles]

    # Calculating mean and max
    agg_exprs.extend([
        F.max(rolling_window_col).alias("max_value"),
        F.mean(rolling_window_col).alias("mean_value")
    ])

    baseline_df = df.groupBy(*grouping_columns).agg(*agg_exprs)

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

def calculate_percentile_rank(grouped_df,
                              all_df,
                              rolling_window_col,
                              percentile_cols: list,
                              grouping_columns
                              ):
    # baseline_columns = [col for col in baseline_df.columns if col.startswith("baseline")]
    # # Check if stretch is 0
    # zero_stretch_cols = [col for col in baseline_columns if 'stretch_0_perc' in col]
    # Find percentile rank
    final_df = None
    for percentile_col in percentile_cols:
        df = grouped_df.filter(F.col(percentile_col)>0)
        df_combined = all_df.select(*grouping_columns, rolling_window_col).unionByName(
            df.select(*grouping_columns, col(percentile_col).alias(rolling_window_col)))

        windowSpec = Window.partitionBy(*grouping_columns).orderBy(rolling_window_col)

        df_with_percentile = df_combined.withColumn("percentile_rank", percent_rank().over(windowSpec)).dropDuplicates()
        join_condition = [df_with_percentile[col] == df[col] for col in grouping_columns]
        join_condition.append(df_with_percentile[rolling_window_col] == df[percentile_col])

        df_new_percentile = df_with_percentile.join(df, join_condition, "inner").select(
            *grouping_columns,  # Select all the grouping columns
            df_with_percentile[rolling_window_col].alias(percentile_col),  # Select percentile column
            df_with_percentile.percentile_rank.alias(f"percentile_{percentile_col}")  # Alias the percentile rank
        )

        # Combine the results for this column with the previous results
        if final_df is None:
            final_df = df_new_percentile
        else:
            final_df = final_df.join(df_new_percentile, grouping_columns, "left")

        return final_df

def plot_percentiles(pandas_df, columns: list, colours: list, alpha: list, bins):
    # Check if bins is a single value or a list for multiple columns
    if isinstance(bins, list) and len(bins) == len(columns):
        multiple_bins = True
    else:
        multiple_bins = False

    for column, colour, transparency in zip(columns, colours, alpha):
        # Use the corresponding bins for each column or the same bins for all columns
        current_bins = bins[columns.index(column)] if multiple_bins else bins
        plt.hist(pandas_df[column], color=colour, alpha=transparency, bins=current_bins, density=True, label=column)

    # Add labels, title, and legend
    plt.xlabel('Values')
    plt.ylabel('Frequency')
    plt.title('Histograms of Percentile Columns')
    plt.legend()

    # Show the plot
    plt.show()




def add_week_number(df, date_column: str, start_date: str):
    df = df.withColumn("date", to_date(col("EVENT_DATE")))
    df = df.withColumn("week_number",
                       (datediff(col("date"), to_date(expr(f"'{start_date}'"),
                        "yyyy-MM-dd")) / 7).cast("int") + 1
                       )
    return df