"""
Persisting Utilities
====================
This module contains utility functions for persisting data.
Functions:
    - create_beam_table: Create delta table in beam if not exists
    - get_table_name: Get table name. Prefixes stg_ if stage table.
    - insert_df_into_table: Insert dataframe into target table, optional delete before insert
    - write_file_to_blob: Write file to blob container
    - write_dataframe_to_blob: Write parquet file to blob (Note should this be a deltatable instead create_beam_table/insert_df_into_table)
    - write_dataframe_to_blob_csv: Write parquet file to blob (Note should this be a deltatable instead create_beam_table/insert_df_into_table)
    - get_absolute_blob_path: Get absolute path to blob file given relative path from container root
    - read_dataframe_from_blob: Read parquet file from blob
    - _get_container: Get container name depending on environment
    - truncate_table: Truncate a stg table
    - read_table: Read data from table
    - read_table_aml: Read data from table
    - uncache_tables: Uncache tables.
    - get_clv_path:
    - register_model: Register model to Azure Model Registry
    - download_model: Download model from AMR
    - load_model: Load model from AMR and return object
    - aml_clone_table_to_blob: Clone beam delta table to blob for reading in AML
    - aml_read_table: Read data from table
    - _aml_get_blob_datastore: Register blob datastore in AML workspace
    - upload_dataset_to_aml: Function to upload data into AML Datasets from Databricks
    - register_dataset_in_aml: Uploads and registers data in AML from Databricks
    - save_chunk: Save dataframe locally
    - register_chunks: Register a chunk of the scoring data
    - save_partition: Save dataframe locally
"""

import os
from typing import Any, Dict, List, Union

from delta.exceptions import MetadataChangedException, ProtocolChangedException
from delta.tables import DeltaTable
from dtaml import constants, utils
from dtaml._internals.databricks import get_dbutils, get_spark
from dtaml.databricks import runtime
from dtaml.logging import get_logger
from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.utils import AnalysisException
import mlflow
from mlflow import MlflowClient

spark = get_spark()
dbutils = get_dbutils()
log = get_logger(__file__)
aml = None


import pickle
from mlflow.tracking import MlflowClient
import logging

logger = logging.getLogger("mlflow_utils")
def get_latest_version(model_name):
    # Example model name: redemption_offer_11111_campaign_240215 
    client = MlflowClient()
    models = client.search_model_versions(f"name='{model_name}'")
    if not models:
        logger.error(f"Model {model_name} was not found in the registry")
        return None
    model_version = max([int(model.version) for model in models])
    run_id = client.get_model_version(model_name, model_version).run_id
    tmp_path = client.download_artifacts(run_id=run_id, path=model_name+"/model.pkl")
    with open(tmp_path,'rb') as f:
        model = pickle.load(f)
    return model




def register_model(model_name, model_object, tags, description):
    metadata = mlflow.sklearn.log_model(
        model_object,
        artifact_path=model_name,
        registered_model_name=model_name
        )
    model_version = metadata.registered_model_version
    client = MlflowClient()
    if tags:
        for k, v in tags.items():
            client.set_model_version_tag(model_name, model_version, k, v)
    client.update_model_version(
        name=model_name,
        version=model_version,
        description=description,
    )


def create_beam_table(
    table_prefix: str,
    lab_database: str,
    factory_database: str,
    sensitivity: str,
    schema: Union[str, DataFrame],
    partition_by: List[str] = [],
    overwrite_table: bool = False,
    assert_equality: bool = False,
    add_load_timestamp: bool = True,
    optimize_write=False,
    auto_compact=False,
) -> str:
    """Create delta table in beam if not exists

    Parameters
    ----------
    table_prefix : str
        Prefix name of table e.g. <team>_<model>_predictions
    lab_database : str
        Name of lab database e.g. <team>_azlab_prod
    factory_database : str
        Name of factory database e.g. retail_analyse_prod
    sensitivity : str
        Name of lab database
        e.g. 'ns': 'nonsensitive', 's': 'sensitive', 'p': 'personal'
    schema : str or DataFrame
        schema definition in string format please use SQL datatypes or pass dataframe to get schema
        e.g. 'col_name1 string, col_name2 integer, col_name3 date'
    partitionBy : List[str]
        columns to create partitions e.g. ['col_name1', 'col_name2']
    overwrite_table : bool
        (Lab env only) overwrites the entire table (used to change schema or partition cols)
        the table must be empty
    assert_equality : bool
        assert schema and partition_by equality if table exists
    add_load_timestamp : bool


    Returns
    -------


    """
    # Concurrent runs can throw ProtocolChangedException when table is empty, keep retrying
    create_success = False
    while not create_success:
        try:
            # Get table name from prefix
            table_name = get_table_name(
                factory_database=factory_database,
                lab_database=lab_database,
                table_prefix=table_prefix,
                sensitivity=sensitivity,
            )
            if add_load_timestamp:
                if isinstance(schema, str):
                    schema = spark.createDataFrame([], schema)
                if "load_timestamp" not in schema.columns:
                    schema = schema.withColumn("load_timestamp", F.current_timestamp())
            kwargs = {
                "delta.autoOptimize.optimizeWrite": "true"
                if optimize_write
                else "false",
                "delta.autoOptimize.autoCompact": "true" if auto_compact else "false",
            }
            kwargs["schema"] = schema
            kwargs["partitionBy"] = partition_by
            kwargs["mode"] = "overwrite" if overwrite_table else "ignore"
            kwargs["assert_equality"] = assert_equality
            if kwargs["mode"] == "overwrite":
                runtime.save_beam_table(
                    df=spark.createDataFrame([], schema=schema.schema),
                    name=table_name,
                    mode="overwrite",
                    overwriteSchema="true",
                    format="delta",
                )
            else:
                runtime.save_beam_table(df=None, name=table_name, **kwargs)
            create_success = True
        except ProtocolChangedException as e:
            log.error(
                f"ProtocolChangedException failed to create database retrying: {str(e)}"
            )
        except AnalysisException as e:
            es = str(e)
            if "not a Delta table" in es:
                log.error(
                    "AnalysisException not a Delta table, checking if table exists"
                )
                db_name, tbl_name = table_name.split(".")
                if spark._jsparkSession.catalog().tableExists(table_name):
                    return table_name
                log.error(f"table does not exist, retrying create {es}")
            else:
                raise e  # re-raise exception
    return table_name


def get_table_name(
    factory_database: str, lab_database: str, table_prefix: str, sensitivity: str
):
    """_summary_

    Parameters
    ----------
    factory_database : str
        _description_
    lab_database : str
        _description_
    table_prefix : str
        _description_
    sensitivity : str
        _description_

    Returns
    -------
    _type_
        _description_
    """
    # Prefix stg_ if we're in lab and factory_database contains analysestg
    if (
        os.environ[constants.ENV_DBK_WORKSPACE] != "etl"
        and "_analysestg_" in factory_database
    ):
        if table_prefix[:4] != "stg_":
            table_prefix = f"stg_{table_prefix}"
    return utils.get_table_name(
        factory_database=factory_database,
        lab_database=lab_database,
        table_prefix=table_prefix,
        sensitivity=sensitivity,
    )


def insert_df_into_table(
    target_tbl_name: str,
    insert_df: DataFrame,
    add_columns=True,
    insert_append=False,
    delete_where: str = None,
    merge_on: str = None,
    repartitions: str = None,
):
    """Insert dataframe into target table, optional delete before insert

    Parameters
    ----------
    target_tbl_name: str :

    insert_df: DataFrame :

    add_columns :
         (Default value = True)
    insert_append :
         (Default value = False)
    delete_where: str :
         (Default value = None)
    merge_on: str :
         (Default value = None)
    repartitions: str :
         (Default value = None)

    Returns
    -------


    """
    if (
        "load_timestamp" not in insert_df.columns
        and spark.sql(f"DESCRIBE TABLE {target_tbl_name}")
        .filter('col_name="load_timestamp"')
        .count()
        == 1
    ):
        insert_df = insert_df.withColumn("load_timestamp", F.current_timestamp())
    if not insert_append:
        df = spark.read.table(target_tbl_name)
        new_cols = []
        for col in df.columns:
            if col not in insert_df.columns:
                insert_df = insert_df.withColumn(col, F.lit(""))
        for col in insert_df.columns:
            if col not in df.columns:
                new_cols.append(col)
        insert_df = insert_df.select(df.columns + new_cols)
    if delete_where:
        spark.sql(f"DELETE FROM {target_tbl_name} WHERE {delete_where}")
    if repartitions:
        try:
            target_tbl_partitionby = list(
                spark.sql(f"SHOW PARTITIONS {target_tbl_name}").columns
            )
            insert_df = insert_df.repartition(repartitions, *target_tbl_partitionby)
        except Exception as e:
            if (
                "SHOW PARTITIONS is not allowed on a table that is not partitioned"
                in str(e)
            ):
                insert_df = insert_df.repartition(repartitions)
            else:
                raise
    merge_success = False
    automerge_init_val = spark.conf.get(
        "spark.databricks.delta.schema.autoMerge.enabled"
    )
    if add_columns:
        spark.conf.set("spark.databricks.delta.schema.autoMerge.enabled", "true")
    else:
        spark.conf.set("spark.databricks.delta.schema.autoMerge.enabled", "false")
    if merge_on:
        while not merge_success:
            try:
                deltaTable = DeltaTable.forName(spark, target_tbl_name)
                (
                    deltaTable.alias("tgt")
                    .merge(insert_df.alias("src"), merge_on)
                    .whenNotMatchedInsertAll()
                    .execute()
                )
                merge_success = True
            except MetadataChangedException as e:
                log.error(f"MetadataChangedException retrying merge: {e}")
    else:
        while not merge_success:
            try:
                if insert_append:
                    insert_df.write.mode("append").saveAsTable(target_tbl_name)
                else:
                    insert_df.write.insertInto(target_tbl_name)
                merge_success = True
            except MetadataChangedException as e:
                log.error(f"MetadataChangedException retrying merge: {e}")
    spark.conf.set(
        "spark.databricks.delta.schema.autoMerge.enabled", automerge_init_val
    )


def write_file_to_blob(
    file_path: str,
    file_data: str,
    overwrite: bool,
    containers: Dict,
    errorifexists: str = True,
):
    """Write file to blob container

    Parameters
    ----------
    file_path: str :

    file_data: str :

    overwrite: bool :

    containers: Dict :

    errorifexists: str :
         (Default value = True)

    Returns
    -------


    """
    assert file_path[0] == "/", "file_path must start with /"
    container = _get_container(containers)
    try:
        get_dbutils().fs.put(
            f"dbfs:/mnt/{container}{file_path}", file_data, overwrite=overwrite
        )
    except Exception as e:
        if errorifexists or "FileAlreadyExistsException" not in e.args[0]:
            raise e


def write_dataframe_to_blob(
    df: DataFrame, relative_path: str, containers: Dict, **kwargs: dict
):
    """Write parquet file to blob (Note should this be a deltatable instead create_beam_table/insert_df_into_table)

    Parameters
    ----------
    df: DataFrame :

    relative_path: str :

    containers: Dict :

    **kwargs: dict :


    Returns
    -------


    """
    abs_path = get_absolute_blob_path(relative_path, containers)
    df.write.parquet(path=abs_path, **kwargs)
    return abs_path


def write_dataframe_to_blob_csv(
    df: DataFrame, relative_path: str, file_name: str, containers: Dict, **kwargs: dict
):
    """Write parquet file to blob (Note should this be a deltatable instead create_beam_table/insert_df_into_table)

    Parameters
    ----------
    df: DataFrame :

    relative_path: str :

    containers: Dict :

    **kwargs: dict :


    Returns
    -------


    """
    abs_path = get_absolute_blob_path(relative_path, containers)
    df_pd = df.toPandas()
    df_pd.to_csv(f"/dbfs{abs_path}/{file_name}", header=True, index=False, **kwargs)

    return abs_path


def get_absolute_blob_path(relative_path: str, containers: Dict):
    """Get absolute path to blob file given relative path from container root

    Parameters
    ----------
    relative_path : str
        relative path to parquet
    containers : dtaml.config.Config
        container names in config as follows
        containers:
        dev: 'dev_container'
        ppd: 'ppd_container'

    Returns
    -------


    """
    container = _get_container(containers)
    return f"/mnt/{container}/{relative_path}"


def read_dataframe_from_blob(relative_path: str, containers: Dict) -> DataFrame:
    """Read parquet file from blob

    Parameters
    ----------
    relative_path : str
        relative path to parquet
    containers : dtaml.config.Config
        container names in config as follows
        containers:
        dev: 'dev_container'
        ppd: 'ppd_container'

    Returns
    -------


    """
    return spark.read.parquet(get_absolute_blob_path(relative_path, containers))


def _get_container(containers: Dict) -> str:
    """Get container name depending on environment

    Parameters
    ----------
    containers : dtaml.config.Config
        container names in config as follows
        containers:
        dev: 'dev_container'
        ppd: 'ppd_container'


    Returns
    -------


    """
    container = None
    if os.environ[constants.ENV_ENVIRONMENT] == "dev":
        container = containers.dev
    elif os.environ[constants.ENV_ENVIRONMENT] == "ppd":
        container = containers.ppd
    elif os.environ[constants.ENV_ENVIRONMENT] == "prod":
        container = containers.prod
    assert (
        container is not None
    ), f"env not equal to dev/ppd/prod os.environ[constants.ENV_ENVIRONMENT] = {os.environ[constants.ENV_ENVIRONMENT]}"
    return container


def truncate_table(table_name: str):
    """Truncate a stg table

    Parameters
    ----------
    table_name: str :


    Returns
    -------


    """
    database, tbl_prefix = table_name.split(".")
    assert (
        "_analysestg_" in database or tbl_prefix[:4] == "stg_"
    ), "Table must be a stage table"
    spark.sql(f"TRUNCATE TABLE {table_name}")


def read_table(table_name: str, where: str = ""):
    """Read data from table

    Parameters
    ----------
    table_name : str
        Name of table to read
    where : str

    Returns
    -------


    """
    if where != "":
        where = f"WHERE {where}".replace("([", "(").replace("])", ")")
    return spark.sql(f"""SELECT * FROM {table_name} {where}""")


def read_table_aml(table_name: str, partition_filter: str = ""):
    """Read data from table

    Parameters
    ----------
    table_name : str
        Name of table to read
    partition_filter : str
    Returns
    -------


    """
    pass
    # table_path = dtaml.aml.get_table_path(table_name)
    # return Dataset.Tabular.from_parquet_files(f'{table_path}/{partition_filter}', validate=False)


def uncache_tables(tables: List[Dict], dev_database: str):
    """Uncache tables.
    Allows tables to be uncached from memory if source table refresh exception occurs.

    Parameters
    ----------
    tables: List[Dict] :

    dev_database: str :


    Returns
    -------


    """
    for tbl in tables:
        tbl_name = get_table_name(
            factory_database=tbl.factory_database,
            lab_database=dev_database,
            table_prefix=tbl.prefix,
            sensitivity=tbl.sensitivity,
        )
        spark.sql(f"""UNCACHE TABLE IF EXISTS {tbl_name}""")


def get_clv_path(clv_factory_dir: str, clv_mount: Dict):
    return clv_factory_dir.replace(
        "{clv_mount}", clv_mount[os.environ[constants.ENV_ENVIRONMENT]]
    )
def download_model(
    model_name, target_dir=None, tags=None, properties=None, version=None, exist_ok=True
):
    raise NotImplementedError("download_model is not implemented")



def save_partition(
    table_name: str,
    campaign_type_id: int,
    campaign: str,
    test_flag: int,
    partition_name: str,
    partition_value: Any,
    target_dir: str = None,
    drop_columns: List = None,
    enable_full_where_clause: bool = False,
) -> str:
    """Save dataframe locally

    Parameters
    ----------
    scoring_table_name : str
        Name of the scoring table
    campaign_type_id : int
        Campaign type id ('tmo': 1, 'fpo': 2, 'fbo': 3)
    campaign : str
        campaign start date
    test_flag : int
        test group flag
    partition_name: str
        Name of the partition to be considered
    partition_value : int
        data partition to be saved
    target_dir : str
        Temp folder inside tmp where chunk data is to be stored, by default None
    drop_columns : List
        List of columns names to be dropped
    enable_full_where_clause : bool
        Enable where clause filtering campaign and campaign_type_id while reading table
        else filter only on the partition, by default False
    Returns
    -------
    str
        Returns the temp path where the chunk data is stored

    """
    if target_dir:
        tmp_folder = f"/tmp/{target_dir}/{partition_name}={partition_value}"
    else:
        tmp_folder = f"/tmp/scoring/{partition_name}={partition_value}"
    os.makedirs(tmp_folder, exist_ok=True)
    log.info(f"Reading data from: {table_name}")
    if enable_full_where_clause:
        data = read_table(
            table_name=table_name,
            where=f"campaign_type_id={campaign_type_id} and campaign={campaign} and test_flag={test_flag} and {partition_name}={partition_value}",
        )
    else:
        data = read_table(
            table_name=table_name, where=f"{partition_name}={partition_value}"
        )
    log.info(
        f"Converting sparks dataframe to pandas df for {partition_name}: {partition_value}"
    )
    df_pd = data.toPandas()
    log.info(f"Dropping unused cols: {drop_columns}")
    df_pd.drop(
        columns=drop_columns,
        inplace=True,
    )
    save_fp = os.path.join(tmp_folder, f"{partition_name}_{partition_value}.parquet")
    log.info(f"Saving parquet to {save_fp}")
    df_pd.to_parquet(save_fp, engine="pyarrow")
    del df_pd
    return tmp_folder
