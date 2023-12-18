# import os

# from dtaml.config import load_section_config
import os
import yaml
from dtaml.config import load_section_config, Config


def load_config(section, file_name="config.yaml"):
    return load_section_config(
        file_path=os.path.join(os.path.dirname(__file__), file_name),
        section=section,
        use_databricks=False,
    )


def load_config_databricks(section, file_name="config.yaml"):
    return load_section_config(
        file_path=os.path.join(os.path.dirname(__file__), file_name),
        section=section,
        use_databricks=True,
    )

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
    config = load_section_config(
        file_path=file_path,
        section=campaign_type,
    )
    return config