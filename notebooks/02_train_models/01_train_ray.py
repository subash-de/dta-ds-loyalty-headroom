# Databricks notebook source
# MAGIC %run ../setup/bootstrap

# COMMAND ----------

dbutils.widgets.text("seg", "{}", "")

# COMMAND ----------

from datetime import datetime, timedelta
import seaborn as sns
from dtaml.logging import get_logger

import customer_headroom.utils.persist_utils as persist_utils
from customer_headroom.modelling.data_process import DataProcessor
from customer_headroom.modelling.fit import build_recommender

sns.set(style="whitegrid")
logger = get_logger("customer-headroom")

# COMMAND ----------

from dtaml.databricks.runtime import Notebook, run_notebooks

# COMMAND ----------

dbutils.widgets.text("seg_list", "[]", "")

# COMMAND ----------

seg_list = eval(dbutils.widgets.get("seg_list"))

# COMMAND ----------

if seg_list == []:
    dbutils.notebook.exit(True)
else:
    logger.info(f"seg_list: {seg_list}")

# COMMAND ----------

# seg_list = [
#         {"campaign": 20231212, "experian_hh_composition": "Cat_U", "segmentation": 1},
#         {"campaign": 20231212, "experian_hh_composition": "Cat_00", "segmentation": 5},
#         {"campaign": 20231212, "experian_hh_composition": "Cat_00", "segmentation": 3},
#         {"campaign": 20231212, "experian_hh_composition": "Cat_00", "segmentation": 0},
#         {"campaign": 20231212, "experian_hh_composition": "Cat_00", "segmentation": 1},
#         {"campaign": 20231212, "experian_hh_composition": "Cat_00", "segmentation": 6},
#         {"campaign": 20231212, "experian_hh_composition": "Cat_U", "segmentation": 3},
#         {"campaign": 20231212, "experian_hh_composition": "Cat_05", "segmentation": 2},
#         {"campaign": 20231212, "experian_hh_composition": "Cat_U", "segmentation": 6},
#         {"campaign": 20231212, "experian_hh_composition": "Cat_00", "segmentation": 2},
#         {"campaign": 20231212, "experian_hh_composition": "Cat_00", "segmentation": 4},
#         {"campaign": 20231212, "experian_hh_composition": "Cat_U", "segmentation": 0},
#         {"campaign": 20231212, "experian_hh_composition": "Cat_05", "segmentation": 5},
#         {"campaign": 20231212, "experian_hh_composition": "Cat_01", "segmentation": 0},
#         {"campaign": 20231212, "experian_hh_composition": "Cat_05", "segmentation": 0},
#         {"campaign": 20231212, "experian_hh_composition": "Cat_05", "segmentation": 4},
#         {"campaign": 20231212, "experian_hh_composition": "Cat_01", "segmentation": 5},
#         {"campaign": 20231212, "experian_hh_composition": "Cat_01", "segmentation": 4},
#         {"campaign": 20231212, "experian_hh_composition": "Cat_05", "segmentation": 6},
#         {"campaign": 20231212, "experian_hh_composition": "Cat_03", "segmentation": 5},
#         {"campaign": 20231212, "experian_hh_composition": "Cat_U", "segmentation": 5},
#         {"campaign": 20231212, "experian_hh_composition": "Cat_01", "segmentation": 3},
#         {"campaign": 20231212, "experian_hh_composition": "Cat_01", "segmentation": 1},
#         {"campaign": 20231212, "experian_hh_composition": "Cat_03", "segmentation": 3},
#         {"campaign": 20231212, "experian_hh_composition": "Cat_03", "segmentation": 0},
#         {"campaign": 20231212, "experian_hh_composition": "Cat_05", "segmentation": 1},
#         {"campaign": 20231212, "experian_hh_composition": "Cat_01", "segmentation": 2},
#         {"campaign": 20231212, "experian_hh_composition": "Cat_03", "segmentation": 1},
#         {"campaign": 20231212, "experian_hh_composition": "Cat_U", "segmentation": 4},
#         {"campaign": 20231212, "experian_hh_composition": "Cat_U", "segmentation": 2},
#         {"campaign": 20231212, "experian_hh_composition": "Cat_02", "segmentation": 3},
#         {"campaign": 20231212, "experian_hh_composition": "Cat_02", "segmentation": 5},
#         {"campaign": 20231212, "experian_hh_composition": "Cat_02", "segmentation": 4},
#         {"campaign": 20231212, "experian_hh_composition": "Cat_03", "segmentation": 6},
#         {"campaign": 20231212, "experian_hh_composition": "Cat_02", "segmentation": 0},
#         {"campaign": 20231212, "experian_hh_composition": "Cat_02", "segmentation": 2},
#         {"campaign": 20231212, "experian_hh_composition": "Cat_02", "segmentation": 6},
#         {"campaign": 20231212, "experian_hh_composition": "Cat_08", "segmentation": 2},
#         {"campaign": 20231212, "experian_hh_composition": "Cat_01", "segmentation": 6},
#         {"campaign": 20231212, "experian_hh_composition": "Cat_05", "segmentation": 3},
#         {"campaign": 20231212, "experian_hh_composition": "Cat_03", "segmentation": 2},
#         {"campaign": 20231212, "experian_hh_composition": "Cat_04", "segmentation": 0},
#         {"campaign": 20231212, "experian_hh_composition": "Cat_08", "segmentation": 1},
#         {"campaign": 20231212, "experian_hh_composition": "Cat_04", "segmentation": 1},
#         {"campaign": 20231212, "experian_hh_composition": "Cat_03", "segmentation": 4},
#         {"campaign": 20231212, "experian_hh_composition": "Cat_02", "segmentation": 1},
#         {"campaign": 20231212, "experian_hh_composition": "Cat_08", "segmentation": 5},
#         {"campaign": 20231212, "experian_hh_composition": "Cat_08", "segmentation": 3},
#         {"campaign": 20231212, "experian_hh_composition": "Cat_10", "segmentation": 0},
#         {"campaign": 20231212, "experian_hh_composition": "Cat_07", "segmentation": 2},
#         {"campaign": 20231212, "experian_hh_composition": "Cat_07", "segmentation": 0},
#         {"campaign": 20231212, "experian_hh_composition": "Cat_06", "segmentation": 0},
#         {"campaign": 20231212, "experian_hh_composition": "Cat_08", "segmentation": 4},
#         {"campaign": 20231212, "experian_hh_composition": "Cat_08", "segmentation": 0},
#         {"campaign": 20231212, "experian_hh_composition": "Cat_07", "segmentation": 5},
#         {"campaign": 20231212, "experian_hh_composition": "Cat_09", "segmentation": 0},
#         {"campaign": 20231212, "experian_hh_composition": "Cat_04", "segmentation": 6},
#         {"campaign": 20231212, "experian_hh_composition": "Cat_10", "segmentation": 2},
#         {"campaign": 20231212, "experian_hh_composition": "Cat_07", "segmentation": 1},
#         {"campaign": 20231212, "experian_hh_composition": "Cat_10", "segmentation": 4},
#         {"campaign": 20231212, "experian_hh_composition": "Cat_04", "segmentation": 4},
#         {"campaign": 20231212, "experian_hh_composition": "Cat_10", "segmentation": 1},
#         {"campaign": 20231212, "experian_hh_composition": "Cat_04", "segmentation": 5},
#         {"campaign": 20231212, "experian_hh_composition": "Cat_07", "segmentation": 4},
#         {"campaign": 20231212, "experian_hh_composition": "Cat_10", "segmentation": 5},
#         {"campaign": 20231212, "experian_hh_composition": "Cat_04", "segmentation": 3},
#         {"campaign": 20231212, "experian_hh_composition": "Cat_07", "segmentation": 3},
#         {"campaign": 20231212, "experian_hh_composition": "Cat_11", "segmentation": 0},
#         {"campaign": 20231212, "experian_hh_composition": "Cat_07", "segmentation": 6},
#         {"campaign": 20231212, "experian_hh_composition": "Cat_08", "segmentation": 6},
#         {"campaign": 20231212, "experian_hh_composition": "Cat_04", "segmentation": 2},
#         {"campaign": 20231212, "experian_hh_composition": "Cat_10", "segmentation": 6},
#         {"campaign": 20231212, "experian_hh_composition": "Cat_10", "segmentation": 3},
#         {"campaign": 20231212, "experian_hh_composition": "Cat_06", "segmentation": 4},
#         {"campaign": 20231212, "experian_hh_composition": "Cat_06", "segmentation": 1},
#         {"campaign": 20231212, "experian_hh_composition": "Cat_06", "segmentation": 2},
#         {"campaign": 20231212, "experian_hh_composition": "Cat_06", "segmentation": 3},
#     ]

# COMMAND ----------

type(seg_list)

# COMMAND ----------


notebooks = [
  Notebook(
    name=f"headroomX_train_campaign_{seg['campaign']}_exp_{seg['experian_hh_composition']}_seg_{seg['segmentation']}",
    path="./02_train_single",
    args={
      "seg": str(seg),
    }
  )
  for seg in seg_list
]

# COMMAND ----------

notebooks[0].args

# COMMAND ----------

run_notebooks(
  notebooks,
  parallel=30,
  progress=30
)

# COMMAND ----------

dbutils.notebook.exit(True)
