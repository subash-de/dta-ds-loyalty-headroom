# COMMAND ----------

# MAGIC %run ./version

# COMMAND ----------

devops_token = dbutils.secrets.get("dta-eun-kv-dsc-01", "access-token-devops-artifacts")
pip_url = PIP_URL.format(token=devops_token)

%pip install --extra-index-url "{pip_url}" "{PACKAGE_NAME}=={PACKAGE_VERSION}"

# COMMAND ----------
from dtaml.databricks import get_all_widgets
from customer_headroom.config import load_config

widgets = get_all_widgets()
env = widgets.get("environment", 'dev')
config = load_config(env)
print(f'Config used is: \n{config.dumps()}')
