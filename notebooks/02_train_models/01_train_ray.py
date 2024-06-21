# Databricks notebook source
# MAGIC %run ../setup/bootstrap

# COMMAND ----------

dbutils.widgets.text("seg", "{}", "")

# COMMAND ----------

import seaborn as sns
from dtaml.logging import get_logger

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

type(seg_list)

# COMMAND ----------

# //TODO #27 - We need to migrate model training from Dtaml Notebooks to Ray Tune
notebooks = [
    Notebook(
        name=f"headroomX_train_campaign_{seg['campaign']}_exp_{seg['experian_hh_composition']}_seg_{seg['segmentation']}",
        path="./02_train_single",
        args={
            "seg": str(seg),
        },
    )
    for seg in seg_list
]

# COMMAND ----------

notebooks[0].args

# COMMAND ----------

run_notebooks(notebooks, parallel=30, progress=30)

# COMMAND ----------

dbutils.notebook.exit(True)
