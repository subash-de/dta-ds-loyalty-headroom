# import os

# from dtaml.config import load_section_config
import os

import yaml
from dtaml.config import Config, load_section_config


def load_config_campaign_type(file_name: str = "config.yaml") -> Config:
    """Loads config.yaml file and selects the appropriate section based on the value of campaign type
    Also formats the pipeline names

    Parameters
    ----------
    file_name : str, optional
        name of configuration file that is inside of the project folder, by default "config.yaml"
    file_name: str :
         (Default value = "config.yaml")

    Returns
    -------
    Config
        configuration class


    """
    file_path = os.path.join(os.path.dirname(__file__), file_name)
    with open(file_path, "r") as stream:
        try:
            config_yml = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            raise exc
    campaign_type = config_yml["comms_flag"]
    config = load_config(
        file_name=file_path,
        section=campaign_type,
    )
    return config


from dtaml.config import get_env_map
from dtaml.databricks.runtime import get_all_widgets, get_spark
from dtaml.pipeline.steps import init_jinja
from dtaml.utils.table import set_default_dbs


def load_config(section, file_name="config.yaml"):
    config = load_section_config(
        file_path=os.path.join(os.path.dirname(__file__), file_name),
        section=section,
    )
    config.envs = get_env_map()
    config.render(config.envs)
    return config


def set_spark_config():
    spark = get_spark()
    for k, v in {
        "spark.sql.sources.partitionOverwriteMode": "dynamic",
        "spark.databricks.delta.optimizeWrite.enabled": "true",
        "spark.databricks.delta.autoCompact.enabled": "true",
    }.items():
        spark.conf.set(k, v)


# TODO check if we can get this to work
def initialize():
    widgets = get_all_widgets()
    env = widgets.get("environment", "dev")

    config = load_config(env)
    init_jinja(config.package_name)
    set_spark_config()
    set_default_dbs(
        config.factory_database, config.lab_database, config.staging_database
    )

    return config
