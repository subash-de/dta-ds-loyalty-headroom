# Databricks notebook source
# MAGIC %run ../setup/bootstrap

# COMMAND ----------

dbutils.widgets.text(
    "segs", "[]", ""
)  # [{"campaign": 20241006, "experian_hh_composition": "Cat_00", "segmentation": 0},]

# COMMAND ----------

import ray
from ray.util.spark import setup_ray_cluster, shutdown_ray_cluster

import mlflow
from pyspark.sql.types import StructType, StructField, StringType, MapType
from pyspark.sql.functions import col, size

from datetime import datetime, timedelta
import customer_headroom.utils.persist_utils as persist_utils
from customer_headroom.modelling.data_process import DataProcessor
from customer_headroom.modelling.fit import build_recommender
from dtaml.logging import get_logger

logger = get_logger("customer-headroom")

# COMMAND ----------

def get_date(date: str) -> int:
    """
    Convert the given date to an integer in the format YYYYMMDD. If the date is 'today', return the current date.

    Parameters:
    date (str): The input date as a string.

    Returns:
    int: The date as an integer in the format YYYYMMDD.
    """
    if str(date).lower() == "today":
        date = datetime.now().strftime("%Y%m%d")
    return int(date)


def get_campaign(campaign: str, etl_date: str) -> int:
    """
    Determine the campaign date. If the campaign is not specified, use the ETL date.

    Parameters:
    campaign (str): The campaign date as a string.
    etl_date (str): The ETL date as a string.

    Returns:
    int: The campaign date as an integer in the format YYYYMMDD.
    """
    if (campaign == "{campaign}") or (campaign == ""):
        campaign = get_date(etl_date)
    return campaign

# COMMAND ----------

debug = False
if debug:
    config.dates.etl_date = config.debug.tables.etl_date
    config.dates.lookback_days = config.debug.tables.lookback_days

campaign = get_campaign(config.dates.upcoming_campaign, config.dates.etl_date)

last_registration_date = int(
    (
        datetime.strptime(str(campaign), config.dates.date_format)
        - timedelta(days=config.dates.lookback_days_registration)
    ).strftime(config.dates.date_format)
)

logger.info(
    f"""
config.dates: {config.dates}
campaign: {campaign}
last_registration_date: {last_registration_date}
"""
)

# COMMAND ----------

segs = eval(dbutils.widgets.get("segs"))

config_fr = config["fit_rec"]
partitionByList = config_fr["partitionByList"]

if len(segs) == 0:
    etl_data_tbl_name = persist_utils.get_table_name(
        factory_database=config.factory_database,
        lab_database=config.lab_database,
        table_prefix=config_fr.etl_data_tbl.prefix,
        sensitivity=config.sensitivity,
    )
    logger.info(f"etl_data_tbl_name: {etl_data_tbl_name}")
    segs = (
        persist_utils.read_table(
            table_name=etl_data_tbl_name, where=f"campaign = {campaign}"
        )
        .select(*partitionByList)
        .distinct()
        .toPandas()
        .to_dict("records")
    )

logger.info(f"segs: {segs}")

seg_data_dict = {}

for seg in segs:
    seg_ext = [f"({k}='{seg[k]}')" for k in partitionByList]
    ext_str = "_".join([str(seg[k]) for k in partitionByList])

    seg_etl_data_tbl = persist_utils.read_table(
        table_name=etl_data_tbl_name, where=" and ".join(seg_ext)
    )

    seg_data_dict[ext_str] = seg_etl_data_tbl.toPandas()

logger.info(f"Training dataset is ready for all segments!")

# COMMAND ----------

ctx = dbutils.notebook.entry_point.getDbutils().notebook().getContext()
host_url = ctx.apiUrl().get()
host_token = ctx.apiToken().get()

runtime_env = {
    "env_vars": {
        "MLFLOW_TRACKING_URI": "databricks",
        "DATABRICKS_HOST": host_url,
        "DATABRICKS_TOKEN": host_token,
    }
}

setup_ray_cluster(
    max_worker_nodes=ray.util.spark.MAX_NUM_WORKER_NODES,
    num_cpus_worker_node=6,
    num_cpus_head_node=0,
    num_gpus_worker_node=0,
    num_gpus_head_node=0,
)

# COMMAND ----------

ray.init(runtime_env=runtime_env)

# COMMAND ----------

ray.cluster_resources()

# COMMAND ----------

@ray.remote
def run_fit_rec(seg, config, seg_data, run_id):
    mlflow.set_registry_uri("databricks")
    mlflow.set_tracking_uri("databricks")

    config_fr = config["fit_rec"]
    partitionByList = config_fr["partitionByList"]
    seg_ext = [f"({k}='{seg[k]}')" for k in partitionByList]
    ext_str = "_".join([str(seg[k]) for k in partitionByList])

    model_tags = {**config.get("model_tags", {}), **{"campaign": campaign}}

    err = {}

    with mlflow.start_run(run_id=run_id, nested=True) as parent_run:
        experiment_id = parent_run.info.experiment_id

        with mlflow.start_run(
            experiment_id=experiment_id, nested=True
        ) as data_process_manager_run:

            data_process_manager = DataProcessor(
                feature_col=config_fr["feature_col"],
                item_id=config_fr["item_id"],
                user_id=config_fr["user_id"],
                lognorm=config_fr["lognorm"],
                line_format=config_fr["line_format"],
                min_lim=config_fr["min_lim"],
                max_lim=config_fr["max_lim"],
            )

            seg_data[config_fr["feature_col"]] = seg_data[
                config_fr["feature_col"]
            ].astype(float)
            rec_data = data_process_manager.get(seg_data)
            logger.info(f"{seg}: Recommender Data Created")

            data_process_manager_name = (
                config_fr.data_processor_name + "_{ext}" + "{model_prefix}"
            ).format(
                campaign=campaign, ext=ext_str, model_prefix=config_fr["model_prefix"]
            )
            logger.info(
                f"{seg}: Saving Preprocessor obj={data_process_manager}, name={data_process_manager_name}"
            )

            try:
                mlflow.sklearn.log_model(
                    data_process_manager,
                    artifact_path=data_process_manager_name,
                    registered_model_name=data_process_manager_name,
                )
            except Exception as e:
                err[data_process_manager_name] = str(e)
            
            mlflow.log_metric("rating_scale_min", data_process_manager.rating_scale[0])
            mlflow.log_metric("rating_scale_max", data_process_manager.rating_scale[1])

        with mlflow.start_run(experiment_id=experiment_id, nested=True) as rec_algo_run:

            logger.info(f"{seg}: Build Recommender")

            rec_algo, fit_params = build_recommender(
                X=rec_data,
                method=config_fr["method"],
                params=config_fr["params"],
                param_grid=config_fr["param_grid"],
            )

            rec_name = (config_fr.rec_name + "_{ext}" + "{model_prefix}").format(
                ext=ext_str, model_prefix=config_fr["model_prefix"]
            )
            logger.info(f"{seg}: Saving Recommender obj={rec_algo}, name={rec_name}")

            try:
                mlflow.sklearn.log_model(
                    rec_algo, artifact_path=rec_name, registered_model_name=rec_name
                )
            except Exception as e:
                err[rec_name] = str(e)
            
            for param, value in fit_params.items():
                mlflow.log_param(param, value)

        with mlflow.start_run(
            experiment_id=experiment_id, nested=True
        ) as fit_params_run:

            param_name = (config_fr.param_name + "_{ext}" + "{model_prefix}").format(
                ext=ext_str, model_prefix=config_fr["model_prefix"]
            )
            logger.info(f"{seg}: Saving Parameters obj={rec_algo}, name={param_name}")

            try:
                mlflow.sklearn.log_model(
                    fit_params,
                    artifact_path=param_name,
                    registered_model_name=param_name,
                )
            except Exception as e:
                err[param_name] = str(e)

            for param, value in fit_params.items():
                mlflow.log_param(param, value)

    return (model_tags, seg, data_process_manager_name, rec_name, param_name, run_id, err)


logger.info("Begin Preprocessing dataset.")

config_fr = config["fit_rec"]
partitionByList = config_fr["partitionByList"]

main_expt_path = config_fr["main_expt_path"]
main_expt_name = config_fr["main_expt_name"]
experiment = f"{main_expt_path}/{main_expt_name}"
logger.info(f"experiment: {experiment}")

try:
    experiment_id = mlflow.create_experiment(name=f"/Workspace{experiment}")
except:
    experiment_id = mlflow.get_experiment_by_name(name=experiment).experiment_id

with mlflow.start_run(experiment_id=experiment_id) as run:
    results = ray.get(
        [
            run_fit_rec.remote(
                seg,
                config,
                seg_data_dict["_".join([str(seg[k]) for k in partitionByList])],
                run.info.run_id,
            )
            for seg in segs
        ]
    )

# COMMAND ----------

schema = StructType([
    StructField("model_tags", MapType(StringType(), StringType()), True),
    StructField("seg", MapType(StringType(), StringType()), True),
    StructField("data_process_manager_name", StringType(), True),
    StructField("rec_name", StringType(), True),
    StructField("param_name", StringType(), True),
    StructField("run_id", StringType(), True),
    StructField("err", MapType(StringType(), StringType()), True)
])

results_df = spark.createDataFrame(results, schema=schema)
display(results_df)

# COMMAND ----------

if results_df.filter(size(col("err")) > 0).count() > 0:
    raise Exception("Job failed due to errors in the 'err' column.")

# COMMAND ----------

try:
    ray.util.spark.shutdown_ray_cluster()
except:
    ray.shutdown()

# COMMAND ----------

dbutils.notebook.exit(True)
