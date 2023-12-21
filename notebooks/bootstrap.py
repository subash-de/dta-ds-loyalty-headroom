# Databricks notebook source
# MAGIC %run ./version

# COMMAND ----------



import shutil
import subprocess
from pathlib import Path
 

devops_token = dbutils.secrets.get("dta-eun-kv-dsc-01", "access-token-devops-artifacts")
pip_url = PIP_URL.format(token=devops_token)
 
%pip config set global.extra-index-url "{pip_url}"
if PACKAGE_SOURCE=="repos":
    try:
      lib_root = dbutils.widgets.get('folder')
    except:
      lib_root = '..'
    py_root = (Path.cwd() / lib_root).resolve()
    egg_base = ('/tmp' / py_root.relative_to('/'))
    shutil.rmtree(egg_base, ignore_errors=True)
    egg_base.mkdir(parents=True, exist_ok=True)
    subprocess.run(f"python '{py_root}/setup.py' egg_info --egg-base '{egg_base}'", shell=True, check=True)
    req_file = list(egg_base.glob('**/requires.txt'))[0]
    %pip install -r "{req_file}"
else:
    %pip install "{PACKAGE_NAME}=={PACKAGE_VERSION}"

    

# COMMAND ----------

'''
devops_token = dbutils.secrets.get("dta-eun-kv-dsc-01", "access-token-devops-artifacts")
pip_url = PIP_URL.format(token=devops_token).replace('%40prerelease', '')

%pip config set global.extra-index-url "{pip_url}"

if PACKAGE_SOURCE=="repos":
    try:
        %pip install -e "{'/Workspace'+'/'.join(dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get().split('/', 4)[:4])}"
    except:
        pass
else:
    %pip install "{PACKAGE_NAME}=={PACKAGE_VERSION}"

    
'''   


# COMMAND ----------

from dtaml.databricks.runtime import get_all_widgets
from customer_headroom.config import load_config, load_config_campaign_type

widgets = get_all_widgets()
# env = widgets.get("environment", 'dev')
config = load_config_campaign_type()
print(f'Config used is: \n{config.dumps()}')
