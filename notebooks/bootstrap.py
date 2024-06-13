# Databricks notebook source
# MAGIC %run ./version

# COMMAND ----------

from pathlib import Path

devops_token = dbutils.secrets.get("dta-eun-kv-dsc-01", "access-token-devops-artifacts")
pip_url = PIP_URL.format(token=devops_token)

%pip config set global.extra-index-url "{pip_url}"
if PACKAGE_SOURCE=="repos":
    try:
      lib_root = dbutils.widgets.get('folder')
    except:
      lib_root = '..'
    py_root = (Path('/Workspace') / Path(dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get().lstrip('/')).parent / lib_root).resolve()
    %pip install "{py_root}"
else:
    %pip install "{PACKAGE_NAME}=={PACKAGE_VERSION}"

dbutils.library.restartPython()




# COMMAND ----------

from dtaml.databricks.runtime import get_all_widgets

from customer_headroom.config import load_config_campaign_type

widgets = get_all_widgets()
config = load_config_campaign_type()
print(f'Config used is: \n{config.dumps()}')
