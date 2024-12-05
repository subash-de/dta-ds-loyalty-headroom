from pyspark.sql import functions as F
from pyspark.sql import DataFrame, Window
import random
from functools import reduce
from typing import Optional, Literal

def assignment(df: DataFrame,
               user_id: str,
               treatment_ratio: float = 0.5,
               test_cell_split: Optional[dict] = None,
               method: Literal["percentage", "volume", "random"] = "random",
    ) -> DataFrame:
    """
    Assigns customers to test types based on a specified method.

    Parameters:
    df (DataFrame): The input Spark DataFrame with customer data.
    user_id (str): The user id for test cell assignment.
    treatment_ratio (float): The ratio of customers in treatment group
    test_cell_split (dict, optional): The test cell split specifications.
        - For 'random' method, don't have any split specified.
        - For 'percentage' method, values should be percentages (summing up to 1).
        - For 'volume' method, values should be raw counts (summing up to the total number of customers or less).
    method (str): The assignment method, can be 'percentage', 'volume', or 'random'.

    Returns:
    DataFrame: A new Spark DataFrame with each customer assigned to one test type.
    """

    # Validate 'test_cell_split' for 'percentage' or 'volume' methods
    if method in {"percentage", "volume"} and test_cell_split is None:
        raise ValueError("The 'test_cell_split' dictionary is required for 'percentage' and 'volume' methods.")

    # Collect distinct test types in data
    distinct_test_types = [row["test_type"] for row in df.select("test_type").distinct().collect()]
    
    if method != "random":
        # Ensure test types in config match those in data
        if "treatment" in test_cell_split.keys():
            test_types_in_config = set(test_cell_split["treatment"].keys()).intersection(set(test_cell_split["control"].keys()))
        else:
            test_types_in_config = set(test_cell_split.keys())

    # Get distinct customers, shuffle for randomization
    distinct_cust = df.select(user_id).distinct().orderBy(F.rand())
    window_spec = Window.orderBy(F.lit(1))
    distinct_cust = distinct_cust.withColumn("row_num", F.row_number().over(window_spec) - 1)
    distinct_cust.cache()

    total_customers = distinct_cust.count()

    # Define customer counts per test type
    if method == "percentage":
        if sum([pct for _, pct in test_cell_split.items()]) > 1:
            raise ValueError("Sum of specified percentages exceeds 1.")

        test_type_counts = {}
        test_type_counts["treatment"] = {test_type: int(total_customers * pct * treatment_ratio) for test_type, pct in test_cell_split.items()}
        test_type_counts["control"] = {test_type: int(total_customers * pct * (1 - treatment_ratio)) for test_type, pct in test_cell_split.items()}
    elif method == "volume":
        test_type_counts = test_cell_split
        if sum(test_type_counts["treatment"].values()) + sum(test_type_counts["control"].values()) > total_customers:
            raise ValueError("Sum of specified volumes exceeds the number of unique customers.")
    elif method == "random":
        # Equally distribute customers among test types
        num_test_types = len(distinct_test_types)
        test_type_counts = {}
        test_type_counts["treatment"] = {test_type: int(total_customers * treatment_ratio // num_test_types) for test_type in distinct_test_types}
        test_type_counts["control"] = {test_type: int(total_customers * (1 - treatment_ratio) // num_test_types) for test_type in distinct_test_types}
        # Distribute remaining
        for i in range(int((total_customers * treatment_ratio) % num_test_types)):
            test_type_counts["treatment"][distinct_test_types[i]] += 1
        for i in range(int((total_customers * (1 - treatment_ratio)) % num_test_types)):
            test_type_counts["control"][distinct_test_types[i]] += 1

    # Assign customers based on calculated counts
    start_index = 0
    assigned_dfs = []

    for test_group in ["treatment", "control"]:
        for test_type, count in test_type_counts[test_group].items():
            assigned_cust = distinct_cust.filter((F.col("row_num") >= start_index) & (F.col("row_num") < start_index + int(count))) 

            # Filter by test_type in all methods
            test_type_df = (
                df.filter(F.col("test_type") == test_type)
                    .join(assigned_cust.select(user_id), on=user_id, how="inner")
                    .withColumn("test_group", F.lit(test_group))
            )

            assigned_dfs.append(test_type_df)
            start_index += int(count)

    # Combine all assigned DataFrames into one
    final_assigned_df = reduce(DataFrame.unionByName, assigned_dfs)
    # final_assigned_df = assigned_dfs[0]
    # for assigned_df in assigned_dfs[1:]:
    #     final_assigned_df = final_assigned_df.unionByName(assigned_df)

    # Optionally, order the final DataFrame by user_id if needed
    final_assigned_df = final_assigned_df.orderBy(user_id)

    return final_assigned_df

def qa_for_assignment(original_df: DataFrame,
                    allocation_df: DataFrame,
                    user_id: str,
                    treatment_ratio: float = 0.5,
                    test_cell_split: Optional[dict] = None,
                    method: Literal["percentage", "volume", "random"] = "random",
    ) -> bool:
    """
    Quality check for customer allocations to test types.

    Parameters:
    original_df (DataFrame): The input Spark DataFrame with customer prediction data.
    allocation_df (DataFrame): The input Spark DataFrame with test cell allocation data.
    user_id (str): The user id for test cell assignment.
    treatment_ratio (float): The ratio of customers in treatment group
    test_cell_split (dict, optional): The test cell split specifications.
        - For 'random' method, don't have any split specified.
        - For 'percentage' method, values should be percentages (summing up to 1).
        - For 'volume' method, values should be raw counts (summing up to the total number of customers or less).
    method (str): The assignment method, can be 'percentage', 'volume', or 'random'.

    Returns:
    Boolean: An indicator if the quality check passes or not.
    """
    # Check whether there are any customer duplication in the test cell allocation
    cust_appearance_count = allocation_df.groupBy(user_id).count().select("count").distinct().toPandas()["count"]
    if len(cust_appearance_count) != 1 and max(cust_appearance_count) > 1:
        raise ValueError("There are customers who are allocated to test cells multiple times.")

    if method == "random":
        # Check whether there are any customers who are not allocated to any test cells
        if original_df.select(user_id).distinct().count() != allocation_df.select(user_id).distinct().count():
            raise ValueError("There are customers who aren't allocated to any test cells.")
        # Check whether customers are uniformly distributed to test cells
        test_cell_allocation = allocation_df.groupby("test_type").count()
        if test_cell_allocation.select(F.max("count")).collect()[0][0] - test_cell_allocation.select(F.min("count")).collect()[0][0] > 2: # one from control and another one from treatment (in total possibly differ by 2)
            raise ValueError("Customers are not uniformly distributed to test cells.")
        # Check whether treatment vs control split is according to the config
        if round(allocation_df.filter(F.col("test_group") == "treatment").count()/allocation_df.count(), 1) != treatment_ratio:
            raise ValueError("The treatment ratio in the allocation is not correct.")

    elif method == "percentage":
        # Check whether there are any customers who are not allocated to any test cells
        if original_df.select(user_id).distinct().count() - allocation_df.select(user_id).distinct().count() > original_df.select("test_type").distinct().count() * 2:
            raise ValueError("There are customers who aren't allocated to any test cells.")
        # Check whether customers are allocated based on the specified percentage
        for test_type, pct in test_cell_split.items():
            if abs(allocation_df.filter(F.col("test_type") == test_type).count() - int(original_df.select(user_id).distinct().count() * pct)) > 2:
                raise ValueError(f"Test type {test_type} is not allocated the correct number of customers.")
        # Check whether treatment vs control split is according to the config
        if round(allocation_df.filter(F.col("test_group") == "treatment").count()/allocation_df.count(), 1) != treatment_ratio:
            raise ValueError("The treatment ratio in the allocation is not correct.")
    
    elif method == "volume":
        # Check whether customers are allocated based on the specified volume
        for test_group in ["treatment", "control"]:
            for test_type, count in test_cell_split[test_group].items():
                if allocation_df.filter((F.col("test_type") == test_type) &
                                        (F.col("test_group") == test_group)).count() != int(count):
                    raise ValueError(f"Test type {test_type} is not allocated the correct number of customers.")

    return True
