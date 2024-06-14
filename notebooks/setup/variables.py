# Databricks notebook source
import os
from pathlib import Path

file_path = Path(
    dbutils.notebook.entry_point.getDbutils()
    .notebook()
    .getContext()
    .notebookPath()
    .get()
)

# COMMAND ----------

# root folder is two levels up
ROOT_PATH = f"/Workspace{file_path.parents[2]}"
os.environ["ROOT_PATH"] = ROOT_PATH
