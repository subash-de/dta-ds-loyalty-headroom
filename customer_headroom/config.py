import os

from dtaml.config import load_section_config


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
