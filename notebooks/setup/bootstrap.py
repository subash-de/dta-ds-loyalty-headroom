# Databricks notebook source
# MAGIC %run ./version

# COMMAND ----------

from pathlib import Path

devops_token = dbutils.secrets.get("dta-eun-kv-dsc-01", "access-token-devops-artifacts")
pip_url = PIP_URL.format(token=devops_token)

# COMMAND ----------

# MAGIC  %pip config set global.extra-index-url "{pip_url}"

# COMMAND ----------

# MAGIC %run ./variables

# COMMAND ----------



if PACKAGE_SOURCE=="repos":
    %pip install "{ROOT_PATH}"
else:
    %pip install "{PACKAGE_NAME}=={PACKAGE_VERSION}"

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %run ./variables

# COMMAND ----------

from dtaml.databricks.runtime import get_all_widgets

from customer_headroom.config import load_config_campaign_type

widgets = get_all_widgets()
config = load_config_campaign_type(f"{root_path}/config/config.yaml")
print(f'Config used is: \n{config.dumps()}')

# COMMAND ----------


