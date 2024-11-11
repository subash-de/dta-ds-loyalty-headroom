from pyspark.sql import functions as F
from pyspark.sql import DataFrame
from pyspark.sql import Window as W
from pyspark.sql import functions as F
from pyspark.sql.functions import lit 

def random_assignment(df: DataFrame):

  # Assigning random test cells
  distinct_customers = df.select("cust_id").distinct().orderBy(F.rand())
  test_types = df.select("test_type").distinct().collect()
  num_test_types = len(test_types)

  window_spec = W.orderBy(F.lit(1))
  distinct_customers = distinct_customers.withColumn("row_num", F.row_number().over(window_spec))

  test_type_assignments = F.when((distinct_customers["row_num"] % num_test_types) == 0, test_types[0][0])
  for i in range(1, num_test_types):
      test_type_assignments = test_type_assignments.when((distinct_customers["row_num"] % num_test_types) == i, test_types[i][0])

  distinct_customers = distinct_customers.withColumn("test_type", test_type_assignments)

  test_cells_selected = distinct_customers.join(df, on=["cust_id", "test_type"], how="inner").dropDuplicates(["cust_id"])

  return test_cells_selected

