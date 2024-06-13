"""
Persist utils:
* Persist dataframes to delta tables in BEAM
* Read tables in databricks and AML
* Persist files to blob storage
* Read files from blob storage
"""
import os
from datetime import datetime
from typing import Any, Dict, List, Union

import joblib
import pandas as pd
from azureml.core import Dataset, Datastore
from azureml.core.model import Model
from azureml.data.datapath import DataPath
from azureml.exceptions import UserErrorException
from delta.exceptions import MetadataChangedException, ProtocolChangedException
from delta.tables import DeltaTable
from dtaml import constants, utils
from dtaml._internals.databricks import get_dbutils, get_spark
from dtaml.aml import AML
from dtaml.databricks import runtime
from dtaml.logging import get_logger
from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.utils import AnalysisException

spark = get_spark()
dbutils = get_dbutils()
log = get_logger(__file__)
aml = None


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
                    f"AnalysisException not a Delta table, checking if table exists"
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
    """Get table name. Prefixes stg_ if stage table.

    Parameters
    ----------
    factory_database: str :

    lab_database: str :

    table_prefix: str :

    sensitivity: str :


    Returns
    -------

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
        .filter(f'col_name="load_timestamp"')
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


def register_model(
    model_name, model_object, tags=None, properties=None, description=None
):
    """Register model to Azure Model Registry

    Parameters
    ----------
    model_name :

    model_object :

    tags :
         (Default value = None)
    properties :
         (Default value = None)
    description :
         (Default value = None)

    Returns
    -------


    """
    global aml
    if not aml:
        aml = AML()
    model_path = f"/tmp/{model_name}_{datetime.now().strftime('%y%m%d%H%M%S')}.pkl"
    with open(model_path, "wb") as f:
        joblib.dump(model_object, f)
    Model.register(
        aml.get_workspace(),
        model_name=model_name,
        model_path=model_path,
        tags=tags,
        properties=properties,
        description=description,
    )


def download_model(
    model_name, target_dir=None, tags=None, properties=None, version=None, exist_ok=True
):
    """Download model from AMR
    If you want to also load the object from the model file use load_model instead

    Parameters
    ----------
    model_name : str
        model name (As displayed in AMR)
    target_dir : Object
        dir to save model file, if None defaults to /tmp/{model_name}
    tags : list
        filter models by tags e.g. [['upcoming_campaign', '00001'], ['campaign_type', 'tmo']] (Default value = None)
    properties : list
        filter models by properties (Default value = None)
    version : str
        Model version as shown in AMR. None = latest version (Default value = None)
    exist_ok : str
         (Default value = True)

    Returns
    -------


    """
    global aml
    if not aml:
        aml = AML()
    if not target_dir:
        target_dir = f"/tmp/{model_name}/"
    latest = True if version == None else False
    models = Model(workspace=aml.get_workspace(), name=model_name).list(
        aml.get_workspace(),
        name=model_name,
        tags=tags,
        properties=properties,
        latest=latest,
    )
    if latest:
        return models[0].download(target_dir, exist_ok)
    else:
        for model in models:
            if model.version == version:
                return model.download(target_dir, exist_ok)
    raise Exception(f"Model version not found {version}")


def load_model(
    model_name, target_dir=None, tags=None, properties=None, version=None, exist_ok=True
):
    """Load model from AMR and return object
    This method calls download_model and unpickles the model file

    Parameters
    ----------
    model_name : str
        model name (As displayed in AMR)
    target_dir : Object
        dir to save model file, if None defaults to /tmp/{model_name}
    tags : list
        filter models by tags e.g. [['upcoming_campaign', '00001'], ['campaign_type', 'tmo']] (Default value = None)
    properties : list
        filter models by properties (Default value = None)
    version : str
        Model version as shown in AMR. None = latest version (Default value = None)
    exist_ok : str
         (Default value = True)

    Returns
    -------


    """
    model_path = download_model(
        model_name=model_name, target_dir=target_dir, exist_ok=exist_ok
    )
    with open(model_path, "rb") as f:
        return joblib.load(f)


def aml_clone_table_to_blob(table_name: str):
    """Clone beam delta table to blob for reading in AML
    A temporary solution to get data from ADLS beam tables into AML via blob

    Parameters
    ----------
    table_name: str :


    Returns
    -------


    """
    container = _get_container()
    table_prefix = table_name.split(".")[-1]
    spark.sql(
        f"CREATE OR REPLACE TABLE DEEP CLONE {table_name} LOCATION '/mnt/{container}/aml/tables/{table_prefix}'"
    )
    # _aml_register_blob_datastore()


# TODO setup loading
def aml_read_table(table_name: str, datastore: Datastore) -> Dataset:
    """Read table data into Dataset

    Parameters
    ----------
    table_name: str :

    datastore: Datastore :


    Returns
    -------

    """
    pass


# TODO setup get or register datastore, do we want to do this manually?
def _aml_get_blob_datastore(blob_datastore_name: str) -> Datastore:
    """Register blob datastore in AML workspace
    internal method not to be called outside persist_utils.py

    Parameters
    ----------
    blob_datastore_name: str :


    Returns
    -------

    """
    aml = AML()
    ws = aml.get_workspace()
    try:
        blob_datastore = Datastore.get(ws, blob_datastore_name)
        print("Found Blob Datastore with name: %s" % blob_datastore_name)
    except UserErrorException:
        blob_datastore = Datastore.register_azure_blob_container(
            workspace=ws,
            datastore_name=blob_datastore_name,
            account_name=account_name,  # Storage account name
            container_name=container_name,  # Name of Azure blob container
            account_key=account_key,
        )  # Storage account key
        print("Registered blob datastore with name: %s" % blob_datastore_name)


def upload_dataset_to_aml(
    src_dir: str, target_dir: str, datastore: Datastore
) -> Dataset:
    """Function to upload data into AML Datasets from Databricks

    Parameters
    ----------
    src_dir :
        str
    target_dir :
        str
    datastore :
        Datastore
    Returns
    -------
    type
        Dataset: Dataset type object

    """
    ds = Dataset.File.upload_directory(
        src_dir=src_dir,
        target=DataPath(datastore=datastore, path_on_datastore=target_dir),
        overwrite=True,
        show_progress=True,
    )
    return ds


def register_dataset_in_aml(
    src_dir: str,
    target_dir: str,
    datastore_name: str,
    name: str,
    description: str,
    tags: Dict,
    create_new_version: bool = False,
) -> None:
    """Uploads and registers data in AML from Databricks

    Parameters
    ----------
    src_dir : str
        Local source directory for data
    target_dir : str
        Target directory in AML datastore
    datastore : str
        Name of AML datastore
    name : str
        name with which to register dataset
    description : str
        Description of the dataset
    tags : Dict
        Tags for dataset
    create_new_version : bool
        Update dataset version (True) or not (False)
    src_dir: str :

    target_dir: str :

    datastore_name: str :

    name: str :

    description: str :

    tags: Dict :

    create_new_version: bool :
         (Default value = False)

    Returns
    -------
    _type_
        None

    """
    aml = AML()
    current_workspace = aml.get_workspace()
    datastore = Datastore(workspace=current_workspace, name=datastore_name)
    ds = upload_dataset_to_aml(
        src_dir=src_dir, target_dir=target_dir, datastore=datastore
    )
    ds.register(
        workspace=current_workspace,
        name=name,
        description=description,
        tags=tags,
        create_new_version=create_new_version,
    )


def save_chunk(
    scoring_table_name: str,
    campaign_type_id: int,
    campaign: str,
    test_flag: int,
    chunk: int,
    target_dir: str,
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
    chunks : int
        scoring chunk to be saved
    target_dir : str
        Temp folder inside tmp where chunk data is to be stored



    Returns
    -------
    str
        Returns the temp path where the chunk data is stored

    """
    if target_dir:
        tmp_folder = f"/tmp/{target_dir}/chunk={chunk}"
    else:
        tmp_folder = f"/tmp/scoring/chunk={chunk}"
    os.makedirs(tmp_folder, exist_ok=True)
    log.info(f"Reading data from: {scoring_table_name}")
    data = read_table(
        table_name=scoring_table_name,
        where=f"campaign_type_id={campaign_type_id} and campaign={campaign} and test_flag={test_flag} and chunk={chunk}",
    )
    log.info(f"Converting sparks dataframe to pandas df for chunk: {chunk}")
    df_pd = data.toPandas()
    log.info(f"Dropping unused cols")
    df_pd.drop(
        columns=[
            "load_timestamp",
            "test_flag",
            "campaign_type_id",
            "campaign",
            "chunk",
        ],
        inplace=True,
    )
    save_fp = os.path.join(tmp_folder, f"chunk_{chunk}.parquet")
    log.info(f"Saving parquet to {save_fp}")
    df_pd.to_parquet(save_fp, engine="pyarrow")
    return tmp_folder


def register_chunks(
    chunk: int,
    scoring_table_name: str,
    campaign_type_id: int,
    campaign_type: str,
    campaign: str,
    test_flag: int,
    datastore_name: str,
    target_dir: str,
    create_new_version: bool = False,
) -> None:
    """Register a chunk of the scoring data

    Parameters
    ----------
    chunk : int
        scoring chunk to be registered
    scoring_table_name : str
        Name of the scoring table
    campaign_type_id : int
        Campaign type id ('tmo': 1, 'fpo': 2, 'fbo': 3)
    campaign_type : str
        Campaign Type ('tmo', 'fpo', 'fbo')
    campaign : str
        campaign start date
    test_flag : int
        test group flag,
    datastore_name : str
        Name of AML datastore
    target_dir : str
        Path for data upload
    create_new_version: bool :
         (Default value = False)

    Returns
    -------

    """
    folder = save_chunk(
        scoring_table_name=scoring_table_name,
        campaign_type_id=campaign_type_id,
        campaign=campaign,
        test_flag=test_flag,
        chunk=chunk,
        target_dir=target_dir,
    )
    if not (target_dir):
        new_target_dir = (
            f"scoring/{campaign_type}/{campaign}/test_flag={test_flag}/chunk={chunk}"
        )
    else:
        new_target_dir = f"{target_dir}/{campaign_type}/{campaign}/test_flag={test_flag}/chunk={chunk}"
    log.info(f"Target dir: {target_dir}")
    register_dataset_in_aml(
        src_dir=folder,
        target_dir=new_target_dir,
        datastore_name=datastore_name,
        name=f"{campaign_type}_{campaign}_{test_flag}_{chunk}",
        description=f"Registering scoring chunks campaign {campaign}, test group {test_flag}",
        tags={
            "campaign_type": f"{campaign_type}",
            "campaign": f"{campaign}",
            "test_flag": f"{test_flag}",
            "chunk": chunk,
        },
        create_new_version=create_new_version,
    )


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


def upload_partition(
    partition_name: str,
    partition_value: Any,
    table_name: str,
    campaign_type_id: int,
    campaign_type: str,
    campaign: str,
    test_flag: int,
    datastore_name: str,
    target_dir: str,
    drop_columns: List,
    enable_full_where_clause: bool,
) -> None:
    """Upload all data corresponding to the value of a partition

    Parameters
    ----------
    partition_name : str
        Name of the partition
    partition_value: Any
        Value of the partition that needs to be uploaded
    table_name : str
        Name of the scoring table
    campaign_type_id : int
        Campaign type id ('tmo': 1, 'fpo': 2, 'fbo': 3)
    campaign_type : str
        Campaign Type ('tmo', 'fpo', 'fbo')
    campaign : str
        campaign start date
    test_flag : int
        test group flag,
    datastore_name : str
        Name of AML datastore
    target_dir : str
        Path for data upload
    drop_columns : List
        List of columns names to be dropped
    enable_full_where_clause : bool
        Enable where clause filtering campaign and campaign_type_id while reading table
        else filter only on the partition, by default False

    Returns
    -------

    """
    folder = save_partition(
        table_name=table_name,
        campaign_type_id=campaign_type_id,
        campaign=campaign,
        test_flag=test_flag,
        partition_name=partition_name,
        partition_value=partition_value,
        target_dir=target_dir,
        drop_columns=drop_columns,
        enable_full_where_clause=enable_full_where_clause,
    )
    if not (target_dir):
        new_target_dir = f"scoring/{campaign_type}/{campaign}/test_flag={test_flag}/{partition_name}={partition_value}"
    else:
        if test_flag:
            new_target_dir = f"{target_dir}/{campaign_type}/{campaign}/test_flag={test_flag}/{partition_name}={partition_value}"
        else:
            new_target_dir = f"{target_dir}/{campaign_type}/{campaign}/{partition_name}={partition_value}"
    log.info(f"Target dir: {new_target_dir}")
    aml = AML()
    current_workspace = aml.get_workspace()
    datastore = Datastore(workspace=current_workspace, name=datastore_name)
    upload_dataset_to_aml(
        src_dir=folder, target_dir=new_target_dir, datastore=datastore
    )


def save_table(
    table_name: str,
    campaign_type_id: int,
    campaign: str,
    test_flag: int,
    target_dir: str,
) -> str:
    """Save dataframe locally

    Parameters
    ----------
    table_name : str
        Name of the table to be saved
    campaign_type_id : int
        Campaign type id ('tmo': 1, 'fpo': 2, 'fbo': 3)
    campaign : str
        campaign start date
    test_flag : int
        test group flag
    chunks : int
        scoring chunk to be saved
    target_dir : str
        Temp folder inside tmp where chunk data is to be stored

    Returns
    -------
    str
        Returns the temp path where the chunk data is stored

    """
    if target_dir:
        tmp_folder = f"/tmp/{target_dir}/"
    else:
        raise TypeError("Argument cannot be NoneType, please pass a valid string")
    os.makedirs(tmp_folder, exist_ok=True)
    log.info(f"Reading data from: {table_name}")
    data = read_table(
        table_name=table_name,
        where=f"campaign_type_id={campaign_type_id} and campaign={campaign} and test_flag={test_flag}",
    )
    log.info(f"Converting sparks dataframe to pandas df for table: {table_name}")
    df_pd = data.toPandas()
    log.info(f"Dropping unused cols")
    df_pd.drop(
        columns=["load_timestamp", "test_flag", "campaign_type_id", "campaign"],
        inplace=True,
    )
    save_fp = os.path.join(tmp_folder, f"{target_dir}.parquet")
    log.info(f"Saving parquet to {save_fp}")
    df_pd.to_parquet(save_fp, engine="pyarrow")
    return tmp_folder


def register_table(
    table_name: str,
    campaign_type_id: int,
    campaign_type: str,
    campaign: str,
    test_flag: int,
    datastore_name: str,
    target_dir: str,
    create_new_version: bool = False,
) -> None:
    """Register a chunk of the scoring data

    Parameters
    ----------
    table_name : str
        Name of the table to be registered
    campaign_type_id : int
        Campaign type id ('tmo': 1, 'fpo': 2, 'fbo': 3)
    campaign_type : str
        Campaign Type ('tmo', 'fpo', 'fbo')
    campaign : str
        campaign start date
    test_flag : int
        test group flag,
    datastore_name : str
        Name of AML datastore
    target_dir : str
        Path for data upload, also used as name of the dataset
    create_new_version: bool :
         (Default value = False)

    Returns
    -------

    """
    folder = save_table(
        table_name=table_name,
        campaign_type_id=campaign_type_id,
        campaign=campaign,
        test_flag=test_flag,
        target_dir=target_dir,
    )
    if target_dir:
        new_target_dir = (
            f"{target_dir}/{campaign_type}/{campaign}/test_flag={test_flag}"
        )
    else:
        raise TypeError()
    log.info(f"Target dir: {target_dir}")
    register_dataset_in_aml(
        src_dir=folder,
        target_dir=new_target_dir,
        datastore_name=datastore_name,
        name=f"{campaign_type}_{campaign}_{test_flag}_{target_dir}",
        description=f"Registering {target_dir} campaign {campaign}, test group {test_flag}",
        tags={
            "campaign_type": f"{campaign_type}",
            "campaign": f"{campaign}",
            "test_flag": f"{test_flag}",
            "data": {target_dir},
        },
        create_new_version=create_new_version,
    )


def read_registered_data(dataset_name: str) -> pd.DataFrame:
    """Read the registered data using dataset name

    Parameters
    ----------
    dataset_name : str
        Registered dataset name

    Returns
    -------

        pd.Dataframe

    """
    aml = AML()
    ws = aml.get_workspace()
    dataset = Dataset.get_by_name(workspace=ws, name=dataset_name)
    dataset_path = dataset.download(f"/tmp/{dataset_name}/", overwrite=True)[0]
    print(f"Reading data from :  {dataset_path}")
    df = pd.read_parquet(dataset_path, engine="pyarrow")
    return df


def run_notebooks(kwargs={}):
    """Run a notebook given the path and feed other parameters through the kwargs

    Parameters
    ----------
    kwargs : dict, optional
        kwargs should have at least path as a key, by default {}
    """
    dbutils.notebook().run(kwargs.pop("path", None), 7200, kwargs)
