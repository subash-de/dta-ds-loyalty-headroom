from pathlib import Path
import os
import inspect
import pendulum
from dtaml.airflow.dag import MainDAG
from airflow.decorators import task_group, task
from dtaml.airflow.databricks import Cluster, DatabricksTask, DatabricksWorkflow
from dtaml.utils.enums import CodeEnvironment

from typing import Any
task_id_data="build_data"
project_name = "headroom"
def get_current_filename():
    # Get the stack frame
    current_frame = inspect.currentframe()

    # Get the calling frame (where this function was invoked)
    calling_frame = inspect.getouterframes(current_frame)[1]

    # Extract the filename from the calling frame
    file_path = calling_frame.filename

    # Get only the file name without the extension
    file_name_without_extension = os.path.splitext(os.path.basename(file_path))[0]

    return file_name_without_extension
print(get_current_filename())

default_args = {
    "owner": "MLOps Team",
}
import json
dag_params = {
    "env": CodeEnvironment.current().name.lower(),
    "test_mode": False,
    "trigger_date": pendulum.now().strftime("%Y-%m-%d"),
    "task_key_data": f"""ds_{project_name}_{get_current_filename()}__workflow__{task_id_data}""",
    "ray_computes":json.dumps({
        "num_workers": 6,
        "num_cpus_worker_node": 16,
        "num_cpus_head_node": 16
    })
}
# "ray_computes":"""{"num_workers":6,"num_cpus_worker_node":16,"num_cpus_head_node":16}"""

print(dag_params)
dag_kwargs = {
    "project_name": project_name,
    "tags": ["dta-ds-loyalty-headroom", "refactor", "test_cells"],
    "default_args": default_args,
    "params": dag_params,
}




with (MainDAG(**dag_kwargs) as dag):
    clusters = [
        Cluster.new(
            cluster_key="compute_optimised_1",
            runtime=14.3,
            driver_type="Standard_F32s_v2",
            worker_type="Standard_F32s_v2",
            num_workers=15,
        )
    ]



    with DatabricksWorkflow(
            group_id="workflow",
            workspace="loyalty",
            repo_name="dta-ds-loyalty-headroom",
            branch_name="develop",
            job_clusters=clusters,
            # do_xcom_push=True,
            notebook_params={k: "{{ params." + k + " }}" for k in dag_params},
    ) as workflow:



        test_cell_assignment = DatabricksTask(
            task_id="test_cell_assignment",
            notebook_path="notebooks/05_test_cell_assignment/01_test_cell_assignment",
            job_cluster_key="compute_optimised_1"
        )

        qa = DatabricksTask(
            task_id="qa",
            notebook_path="notebooks/06_QA/01_qa",
            job_cluster_key="compute_optimised_1"
        )


        test_cell_assignment >> qa